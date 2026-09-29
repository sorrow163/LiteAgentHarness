from __future__ import annotations

import ctypes
import hashlib
import locale
import os
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from src.core.errors import SandboxUnavailable

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
SE_FILE_OBJECT = 1  #系统文件对象的标识
DACL_SECURITY_INFORMATION = 0x00000004  # 获取可安全对象的安全描述符的dacl

# 对可安全对象的安全描述符的acl的操作
GRANT_ACCESS = 1    # 添加一条类型为 允许 的ACE
REVOKE_ACCESS = 4   # 移除一条匹配的ACE

# 继承掩码
OBJECT_INHERIT_ACE = 0x1    # OI，这条 ACE 会被下面的子对象（文件）继承
CONTAINER_INHERIT_ACE = 0x2 # CI，这条 ACE 会被下面的子容器（目录）继承
INHERIT_FILES_AND_DIRS = OBJECT_INHERIT_ACE | CONTAINER_INHERIT_ACE # 子文件和子目录都继承

# ACE类型
ACCESS_ALLOWED_ACE_TYPE = 0x0   # 类型为 允许

# 对申请获取到的令牌的操作掩码
TOKEN_ASSIGN_PRIMARY = 0x0001   # 拿它当新进程的主令牌
TOKEN_DUPLICATE = 0x0002    # 复制这个令牌
TOKEN_QUERY = 0x0008    # 读令牌里的内容
TOKEN_ADJUST_DEFAULT = 0x0080   # 改令牌的默认 DACL
TOKEN_OPEN_MASK = (TOKEN_ASSIGN_PRIMARY | TOKEN_DUPLICATE | TOKEN_QUERY | TOKEN_ADJUST_DEFAULT)

# 令牌的限制类型
DISABLE_MAX_PRIVILEGE = 0x01    # 取消令牌的特权
LUA_TOKEN = 0x04                # 成为『不提权的受限用户』令牌
WRITE_RESTRICTED = 0x08         # 申请写权限时需要额外校验限制列表
RESTRICTED_TOKEN_FLAGS = DISABLE_MAX_PRIVILEGE | LUA_TOKEN | WRITE_RESTRICTED

TOKEN_GROUPS = 2    # SID集合组(不包括用户 SID)
TOKEN_DEFAULT_DACL = 6  # 令牌内容的默认DACL标识

# SID 属性
SE_GROUP_LOGON_ID = 0xC0000000  # 登录会话SID 标识
WIN_WORLD_SID = 1       # Everyone SID 标识

# 权限掩码
MASK_MODIFY = 0x110156  # MASK_MODIFY 是修改类权限掩码，包括写、追加、改属性、删
FILE_ALL_ACCESS = 0x001F01FF    # 全权：读、写、删、改权限，全都能干

CREATE_SUSPENDED = 0x00000004
#: 环境块是 Unicode 的，就必须带这个标志 —— 否则 `CreateProcessAsUserW` 会按 ANSI 去读，
#: 中文路径直接变乱码。
CREATE_UNICODE_ENVIRONMENT = 0x00000400
STARTF_USESTDHANDLES = 0x00000100

JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS = 9
JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 0x00000008
JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000

# 用于设置句柄信息
HANDLE_FLAG_INHERIT = 0x00000001    # 是否可以被子进程继承

#
GENERIC_READ = 0x80000000       # 只要读权限
FILE_SHARE_READ = 0x00000001    # 允许别人同时读
FILE_SHARE_WRITE = 0x00000002   # 允许别人同时写
OPEN_EXISTING = 3               # 它已经存在，直接开，别新建

WAIT_TIMEOUT = 0x00000102

ACL_SIZE_INFORMATION_CLASS = 2

_READ_CHUNK = 64 * 1024
DEFAULT_MAX_OUTPUT_BYTES = 200_000

#: 能力 SID 的推导分母
_SID_SPAN = 2**30 - 1


# ---------------------------------------------------------------------------
# 结构体
# ---------------------------------------------------------------------------
class SID_AND_ATTRIBUTES(ctypes.Structure):
    _fields_ = [("Sid", ctypes.c_void_p),
                ("Attributes", ctypes.c_uint32)]


class TokenGroups(ctypes.Structure):
    """`TOKEN_GROUPS`（变长数组；`* 1` 只是占位，读的时候要自己重取指针）。"""
    _fields_ = [("GroupCount", ctypes.c_uint32),
                ("Groups", SID_AND_ATTRIBUTES * 1)]


class TOKEN_DEFAULT_DACL_STRUCT(ctypes.Structure):
    _fields_ = [("DefaultDacl", ctypes.c_void_p)]


class TRUSTEE_W(ctypes.Structure):
    _fields_ = [("pMultipleTrustee", ctypes.c_void_p),  #
                ("MultipleTrusteeOperation", ctypes.c_int), #
                ("TrusteeForm", ctypes.c_int),  #
                ("TrusteeType", ctypes.c_int),
                ("ptstrName", ctypes.c_void_p)]


class EXPLICIT_ACCESS_W(ctypes.Structure):
    """48 字节（x64）：mask@0、mode@4、inherit@8、Trustee@16、ptstrName@40。

    Trustee：
        - pMultipleTrustee: 多受托人指针（几乎总是 NULL，我们不用）
        - MultipleTrusteeOperation: 多受托人操作（几乎总是 0）
        - TrusteeForm: 受托人是什么形式：0 = SID、1 = 名字、2 = 对象 GUID
        - TrusteeType: 受托人是什么种类：用户 / 组 / 未知……（只影响显示，不影响判定）
        - ptstrName: 指向受托人的指针

    grfAccessPermissions: 给哪些权限——就是权限掩码
    grfAccessMode: 怎么改——就是动作（GRANT_ACCESS 挂 / REVOKE_ACCESS 撤）
    grfInheritance: 继承标志——就是 OI/CI/NP/IO
    """
    _fields_ = [("grfAccessPermissions", ctypes.c_uint32),
                ("grfAccessMode", ctypes.c_int),
                ("grfInheritance", ctypes.c_uint32),
                ("Trustee", TRUSTEE_W)]


class ACL_SIZE_INFORMATION(ctypes.Structure):
    """
    AceCount: int，有多少条 ace

    AclBytesInUse: int，这份 ace 占据多少字节

    AclBytesFree: int，acl 空间还有多少空余字节
    """
    _fields_ = [("AceCount", ctypes.c_uint32),
                ("AclBytesInUse", ctypes.c_uint32),
                ("AclBytesFree", ctypes.c_uint32)]


class ACE_HEADER(ctypes.Structure):
    _fields_ = [("AceType", ctypes.c_ubyte),
                ("AceFlags", ctypes.c_ubyte),
                ("AceSize", ctypes.c_uint16)]


class ACCESS_ALLOWED_ACE(ctypes.Structure):
    """Header:
        AceType: ACE 的类型，0=允许、1=拒绝、2=审计

        AceFlags: 继承标志
            - OI: 0x01，往下传给子文件
            - CI: 0x02，往下传给子目录
            - NP: 0x04，只传一层，不再往孙子传
            - IO: 0x08，只对孩子有效，对它自己无效
            - I: 0x10，这条是继承来的（系统自动打的标记）
        AceSize: int，acl 空间还有多少空余字节
    Mask: 权限掩码：这条 ACE 给哪些权限

    SidStart: SID 的起点，即相对于这条 ACE 开头"的字节偏移量。

        """
    _fields_ = [("Header", ACE_HEADER),
                ("Mask", ctypes.c_uint32),
                ("SidStart", ctypes.c_uint32)]


class STARTUPINFOW(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_uint32),
        ("lpReserved", ctypes.c_wchar_p),
        ("lpDesktop", ctypes.c_wchar_p),
        ("lpTitle", ctypes.c_wchar_p),
        ("dwX", ctypes.c_uint32),
        ("dwY", ctypes.c_uint32),
        ("dwXSize", ctypes.c_uint32),
        ("dwYSize", ctypes.c_uint32),
        ("dwXCountChars", ctypes.c_uint32),
        ("dwYCountChars", ctypes.c_uint32),
        ("dwFillAttribute", ctypes.c_uint32),
        ("dwFlags", ctypes.c_uint32),
        ("wShowWindow", ctypes.c_uint16),
        ("cbReserved2", ctypes.c_uint16),
        ("lpReserved2", ctypes.c_void_p),
        ("hStdInput", ctypes.c_void_p),
        ("hStdOutput", ctypes.c_void_p),
        ("hStdError", ctypes.c_void_p),
    ]


class PROCESS_INFORMATION(ctypes.Structure):
    _fields_ = [("hProcess", ctypes.c_void_p),
                ("hThread", ctypes.c_void_p),
                ("dwProcessId", ctypes.c_uint32),
                ("dwThreadId", ctypes.c_uint32)]


class SECURITY_ATTRIBUTES(ctypes.Structure):
    """
    nLength: 这个结构体自己多大
    lpSecurityDescriptor: 给新建的对象指定一份名单；填 NULL = 不指定，用默认的
    bInheritHandle: 建出来的句柄能不能被子进程继承
    """
    _fields_ = [("nLength", ctypes.c_uint32),
                ("lpSecurityDescriptor", ctypes.c_void_p),
                ("bInheritHandle", ctypes.c_int)]


class IO_COUNTERS(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in (
        "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
        "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", ctypes.c_uint32),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", ctypes.c_uint32),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", ctypes.c_uint32),
                ("SchedulingClass", ctypes.c_uint32)]


class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t)]


# ---------------------------------------------------------------------------
# 库与签名（惰性加载：`ctypes.WinDLL` 只存在于 Windows，模块顶层 import 会在 Linux 上炸）
# ---------------------------------------------------------------------------
_LIBS: dict[str, ctypes.CDLL] = {}


def _libs() -> dict[str, ctypes.CDLL]:
    if _LIBS:
        return _LIBS
    if sys.platform != "win32":
        raise SandboxUnavailable(
            f"写受限令牌后端只在 Windows 上可用（当前平台 {sys.platform}）")
    # 加载 ernel32.dll 和 advapi32.dll 到当前的 python 进程，并通过实例化的 WinDLL实例 kernel32 和 advapi32 进行操作
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)

    # ctypes.c_void_p: void *, ctypes.c_uint32: undesign long, ctypes.c_int: int, ctypes.c_wchar_p: wchar_t *
    vp, u32, i32, wstr = (ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_wchar_p)
    kernel32.GetCurrentProcess.restype = vp
    kernel32.CloseHandle.argtypes = [vp]
    kernel32.LocalFree.argtypes = [vp]
    kernel32.CreatePipe.argtypes = [ctypes.POINTER(vp), ctypes.POINTER(vp),
                                    ctypes.POINTER(SECURITY_ATTRIBUTES), u32]
    kernel32.SetHandleInformation.argtypes = [vp, u32, u32]
    kernel32.CreateFileW.argtypes = [wstr, u32, u32, ctypes.POINTER(SECURITY_ATTRIBUTES),
                                     u32, u32, vp]
    kernel32.CreateFileW.restype = vp
    kernel32.CreateJobObjectW.argtypes = [vp, wstr]
    kernel32.CreateJobObjectW.restype = vp
    kernel32.SetInformationJobObject.argtypes = [vp, ctypes.c_int, vp, u32]
    kernel32.AssignProcessToJobObject.argtypes = [vp, vp]
    #: 诊断用：判断本进程自己是不是已经在一个 Job 里（`AssignProcessToJobObject` 失败时
    #: 最容易被误判的那个原因，见 `_in_a_job`）。
    kernel32.IsProcessInJob.argtypes = [vp, vp, ctypes.POINTER(ctypes.c_int)]
    kernel32.IsProcessInJob.restype = i32
    kernel32.TerminateJobObject.argtypes = [vp, u32]
    kernel32.TerminateProcess.argtypes = [vp, u32]
    kernel32.ResumeThread.argtypes = [vp]
    #: **必须写 restype**：失败时返回 `0xFFFFFFFF`，而默认的 `c_int` 会把它变成 `-1`，
    #: 于是 `== 0xFFFFFFFF` 这个检查永远不命中 —— "检查失败"本身静默失效。
    kernel32.ResumeThread.restype = u32
    kernel32.WaitForSingleObject.argtypes = [vp, u32]
    kernel32.WaitForSingleObject.restype = u32
    kernel32.GetExitCodeProcess.argtypes = [vp, ctypes.POINTER(u32)]

    advapi32.OpenProcessToken.argtypes = [vp, u32, ctypes.POINTER(vp)]
    advapi32.CreateRestrictedToken.argtypes = [ vp, u32, u32, ctypes.POINTER(SID_AND_ATTRIBUTES),
                                                u32, vp, u32, ctypes.POINTER(SID_AND_ATTRIBUTES), ctypes.POINTER(vp)]
    advapi32.GetTokenInformation.argtypes = [vp, ctypes.c_int, vp, u32, ctypes.POINTER(u32)]
    advapi32.SetTokenInformation.argtypes = [vp, ctypes.c_int, vp, u32]
    advapi32.CreateWellKnownSid.argtypes = [ctypes.c_int, vp, vp, ctypes.POINTER(u32)]
    advapi32.ConvertStringSidToSidW.argtypes = [wstr, ctypes.POINTER(vp)]
    advapi32.ConvertSidToStringSidW.argtypes = [vp, ctypes.POINTER(ctypes.c_wchar_p)]
    advapi32.GetLengthSid.argtypes = [vp]
    advapi32.GetLengthSid.restype = u32
    advapi32.CopySid.argtypes = [u32, vp, vp]
    advapi32.FreeSid.argtypes = [vp]
    kernel32.LocalAlloc.argtypes = [u32, ctypes.c_size_t]
    kernel32.LocalAlloc.restype = vp
    advapi32.GetNamedSecurityInfoW.argtypes = [
        wstr, ctypes.c_int, u32, ctypes.POINTER(vp), ctypes.POINTER(vp),
        ctypes.POINTER(vp), ctypes.POINTER(vp), ctypes.POINTER(vp)]
    advapi32.GetNamedSecurityInfoW.restype = u32
    advapi32.SetNamedSecurityInfoW.argtypes = [wstr, ctypes.c_int, u32, vp, vp, vp, vp]
    advapi32.SetNamedSecurityInfoW.restype = u32
    advapi32.SetEntriesInAclW.argtypes = [u32, ctypes.POINTER(EXPLICIT_ACCESS_W), vp,
                                          ctypes.POINTER(vp)]
    advapi32.SetEntriesInAclW.restype = u32
    advapi32.GetAclInformation.argtypes = [vp, vp, u32, ctypes.c_int]
    advapi32.GetAce.argtypes = [vp, u32, ctypes.POINTER(vp)]
    advapi32.EqualSid.argtypes = [vp, vp]
    advapi32.EqualSid.restype = ctypes.c_int
    # `CreateProcessAsUserW` 在 **advapi32** 里（不在 kernel32 —— 名字容易骗人）。
    # 第 8 个参数（`lpEnvironment`）声明成 `c_wchar_p` 而不是 `c_void_p`：我们要传一个**自己
    # 造的环境块**（把子进程的 TEMP 指到私有临时目录），而 `c_void_p` 不收 Python 字符串。
    # `c_wchar_p` 两种都收：`None`（继承父环境）与字符串（含**内嵌 NUL** 的整块）——
    # ctypes 会按字符串全长开缓冲区并原样拷进去，不会被第一个 `\0` 截断。
    advapi32.CreateProcessAsUserW.argtypes = [
        vp, wstr, ctypes.c_wchar_p, ctypes.POINTER(SECURITY_ATTRIBUTES),
        ctypes.POINTER(SECURITY_ATTRIBUTES), i32, u32, ctypes.c_wchar_p, wstr,
        ctypes.POINTER(STARTUPINFOW), ctypes.POINTER(PROCESS_INFORMATION)]
    advapi32.CreateProcessAsUserW.restype = i32

    _LIBS.update(kernel32=kernel32, advapi32=advapi32)
    return _LIBS


def _h(value) -> int:
    """ctypes 给的"句柄值" → 一个纯 `int`。**`0` 表示"没有句柄"。**
    """
    # 处理逻辑：
    # 如果 value 是空值，则没有句柄，返回0；如果value是一个无类型指针类即 ctypes.c_void_p，则返回该指针保存的地址值，如果没有地址值，应返回 0
    # 最后约定，如果非空且不是 ctypes.c_void_p 类实例，那应该是一个整数值
    # 注：ctypes.c_void_p 对应的指针是 void *
    if value is None:
        return 0
    if isinstance(value, ctypes.c_void_p):
        return value.value or 0
    return int(value)


def _why(prefix: str = "") -> str:
    """槽里取最新的错误码，并用 FormatError 把这个码翻成对应文字。"""
    code = ctypes.get_last_error()
    return f"{prefix}Win32 {code} ({ctypes.FormatError(code).strip()})"

# 非生产路径的函数
def _sid_to_str(sid) -> str:
    out = ctypes.c_wchar_p()    # 创建一个宽字符指针类实例，等价在C中创建一个宽字符指针
    if not _libs()["advapi32"].ConvertSidToStringSidW(sid, ctypes.byref(out)):
        return f"<SID 转字符串失败：{_why()}>"
    text = out.value
    _libs()["kernel32"].LocalFree(ctypes.cast(out, ctypes.c_void_p))
    return text


def _decode_output(raw: bytes) -> str:
    """将二进制数解码成字符串"""
    if not raw:
        return ""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode(locale.getpreferredencoding(False), errors="replace")


def _in_a_job() -> bool:
    """本进程自己是不是已经在一个 Job 里。

    **只用于诊断**，不用于控制流：`AssignProcessToJobObject` 失败时报 `Win32 5`（拒绝访问），
    而最常见的原因就是"父进程已经在一个 Job 里、且那个 Job 不允许嵌套"。
    """
    try:
        inside = ctypes.c_int(0)
        if not _libs()["kernel32"].IsProcessInJob(_libs()["kernel32"].GetCurrentProcess(),
                                                  None, ctypes.byref(inside)):
            return False
        return bool(inside.value)
    except Exception:                                # noqa: BLE001 —— 它只是句诊断
        return False


# ---------------------------------------------------------------------------
# 能力 SID
# ---------------------------------------------------------------------------
def capability_sid(seed: bytes, *, with_counter: bool = False) -> str:
    """路径 → `S-1-4-…` 能力 SID 的**文本形式**。

    授权值 4 是**非唯一授权**：账号库里什么都不建，这个 SID 的权力**完全来自指名它的 ACE**。
    `with_counter=True` 再加一个尾子授权 `1`（用它区分"工作区"与"临时目录"两类身份）。
    """
    digest = hashlib.sha256(seed).digest()
    first = int.from_bytes(digest[0:4], "little") % _SID_SPAN + 1
    second = int.from_bytes(digest[4:8], "little") % _SID_SPAN + 1
    return f"S-1-4-{first}-{second}" + ("-1" if with_counter else "")


def workspace_sid(workspace: str) -> str:
    """根据工作区路径获取工作区能力 SID 的文本。
    """
    return capability_sid(workspace.encode("utf-8"))


def temp_sid(temp_root: str) -> str:
    """临时目录路径 → 临时目录能力 SID 的文本形式。
    """
    return capability_sid(b"temp\x00" + temp_root.encode("utf-8"), with_counter=True)


def _to_sid(text: str) -> ctypes.c_void_p:
    """传入一个 SID 的文本 text， 创建一个 void* 类型的指针变量 sid， 使用 ConvertStringSidToSidW
    将 sid 的文本换变成一个二进制数， 并把这个二进制数的地址赋值给 sid并返回这个指针"""
    sid = ctypes.c_void_p()
    if not _libs()["advapi32"].ConvertStringSidToSidW(text, ctypes.byref(sid)):
        raise SandboxUnavailable(f"造 SID 失败（{text}）：{_why()}")
    return sid


#: `LocalAlloc` 的 `LMEM_FIXED`
_LMEM_FIXED = 0x0040


def _copy_sid(source) -> ctypes.c_void_p:
    """source 是一个 ctypes.c_void_p 类型，是一个指针， 将该指针指向的字节数据拷贝到新分配的内存上，
    并返回指向这块内存的指针
    """
    advapi32, kernel32 = _libs()["advapi32"], _libs()["kernel32"]
    # length 是一个整数（Python int），表示这个 SID 占用多少字节。
    length = advapi32.GetLengthSid(source)
    if not length:
        raise SandboxUnavailable(f"GetLengthSid 失败：{_why()}")
    # 分配一块指定字节长度的内存，返回这个内存的地址，_LMEM_FIXED 代表的意思是直接返回一个可用的地址
    memory = kernel32.LocalAlloc(_LMEM_FIXED, length)
    if not memory:
        raise SandboxUnavailable(f"LocalAlloc 失败：{_why()}")

    # 从 source 那个地址开始，连着读 length 个字节，写到 memory 那个地址开始的地方。
    if not advapi32.CopySid(length, memory, source):
        kernel32.LocalFree(memory)
        raise SandboxUnavailable(f"CopySid 失败：{_why()}")
    return memory


# ---------------------------------------------------------------------------
# ACL：读-合并-写
# ---------------------------------------------------------------------------
def _explicit_access(sid, mask: int, mode: int) -> EXPLICIT_ACCESS_W:
    """创建一条"我想怎么改名单"的完整指令"""
    entry = EXPLICIT_ACCESS_W()
    entry.grfAccessPermissions = mask
    entry.grfAccessMode = mode
    entry.grfInheritance = INHERIT_FILES_AND_DIRS
    entry.Trustee.TrusteeForm = 0        # TRUSTEE_IS_SID
    entry.Trustee.TrusteeType = 0        # TRUSTEE_IS_UNKNOWN
    entry.Trustee.ptstrName = sid
    return entry


def _has_exact_ace(dacl, sid, mask: int) -> bool:
    """DACL 里是否已有"给这个 SID、这个掩码、带 `OI|CI`"的允许 ACE。有就不必再写一遍。
    """
    advapi32 = _libs()["advapi32"]
    size = ACL_SIZE_INFORMATION()

    # 返回 BOOL：非零 = 成功，0 = 失败。
    if not advapi32.GetAclInformation(dacl, # dacl 指针
                                      ctypes.byref(size), # 接收结果的缓冲区地址
                                      ctypes.sizeof(size), # 那个缓冲区占多少字节
                                      ACL_SIZE_INFORMATION_CLASS):  # 要问哪一类信息
        return False
    for index in range(size.AceCount):
        ace_ptr = ctypes.c_void_p()

        # 按索引值获取一条 ace 的地址，赋值给指针 ace_ptr
        if not advapi32.GetAce(dacl, index, ctypes.byref(ace_ptr)):
            continue
        # 将无类型指针ace_ptr转变成ACCESS_ALLOWED_ACE类型的指针，然后获取这个指针指向的结构体变量，即一个ACCESS_ALLOWED_ACE类型的实例
        ace = ctypes.cast(ace_ptr, ctypes.POINTER(ACCESS_ALLOWED_ACE)).contents
        if ace.Header.AceType != ACCESS_ALLOWED_ACE_TYPE:
            continue
        # 这条 ACE 会往下传给子文件和子目录吗
        if (ace.Header.AceFlags & INHERIT_FILES_AND_DIRS) != INHERIT_FILES_AND_DIRS:
            continue
        if ace.Mask != mask:
            continue
        # 获取 sid 二进制数的地址，将地址赋值给 sid_at
        sid_at = ctypes.cast(ctypes.byref(ace, ACCESS_ALLOWED_ACE.SidStart.offset), ctypes.c_void_p)
        # 判断两个sid是否相等
        if advapi32.EqualSid(sid_at, sid):
            return True
    return False


def _apply_ace(path: str, sid, mask: int, mode: int, *, skip_if_present: bool = False) -> bool:
    """给一个路径挂或撤一条 ACE。返回"是否真的改了 ACL"。

    Args:
        path (str): 目标目录路径
        sid (ctypes.c_void_p): sid 二进制数的指针
        mask (int): 调 SetEntriesInAclW 时传的一个参数，表明对目标可安全对象的 ACL 的操作
        skip_if_present (bool): 是否对DACL查重，查重到会跳过写入

    """
    # mode 的值和名字以及具体含义
    # 值	名字	                干什么
    # 0	    NOT_USED_ACCESS	    （占位，不用）	❌
    # 1	    GRANT_ACCESS	    加一条「允许」ACE	✅ 挂授权时
    # 2	    SET_ACCESS	        设为指定权限（覆盖原有的）	❌
    # 3	    DENY_ACCESS	        加一条「拒绝」ACE	❌
    # 4	    REVOKE_ACCESS	    移除匹配的 ACE	✅ 撤授权时
    # 5	    SET_AUDIT_SUCCESS	写 SACL：记成功审计	❌
    # 6	    SET_AUDIT_FAILURE	写 SACL：记失败审计

    advapi32, kernel32 = _libs()["advapi32"], _libs()["kernel32"]

    owner, group, dacl, sacl, descriptor = (ctypes.c_void_p() for _ in range(5))

    # 按路径读出那个对象的安全描述符，并根据 security information 决定返回安全描述符的哪些信息
    # 返回 DWORD：0 = 成功，非 0 = 错误码。
    # 获取的是安全描述符的深拷贝的副本的指针
    result = advapi32.GetNamedSecurityInfoW(
        path,   # 第一个参数：目标可安全对象的路径字符串
        SE_FILE_OBJECT, # 第二个参数：这是什么对象，值为1的 SE_FILE_OBJECT 指的是这是一个系统文件对象
        DACL_SECURITY_INFORMATION,  # 第三个参数：我要读哪部分——这里只要 DACL
        ctypes.byref(owner),    # 其后的参数用于接收出参，第一个出参是 Owner 的 SID 指针
        ctypes.byref(group),    # 第二个出参是Group 的 SID 指针
        ctypes.byref(dacl),     # 第三个是 DACL 的指针
        ctypes.byref(sacl),     # 第四个是 SACL 的指针
        ctypes.byref(descriptor)    # 整个安全描述符的指针，用来释放
    )
    if result != 0:
        raise OSError(f"GetNamedSecurityInfoW 失败：Win32 {result}")

    new_acl = ctypes.c_void_p()
    try:
        if skip_if_present and dacl.value and _has_exact_ace(dacl, sid, mask):
            return False
        # 配置一个命令
        entry = _explicit_access(sid, mask, mode)

        # 根据指令和原有的dacl，和并得到新的dacl
        result = advapi32.SetEntriesInAclW(1, # 有几条指令
                                           ctypes.byref(entry), # 指令地址
                                           dacl, # 现有的 dacl 名单，即前边获取的那份 dacl副本
                                           ctypes.byref(new_acl))   # 出参，新构建出来的 dacl 名单
        if result != 0:
            raise OSError(f"SetEntriesInAclW 失败：Win32 {result}")

        # 将指定的可安全对象的安全描述符的 dacl 替换成前边得到的新的 dacl
        result = advapi32.SetNamedSecurityInfoW(path, # 需要访问的那个可安全对象的路径
                                                SE_FILE_OBJECT, # 它是什么类型
                                                DACL_SECURITY_INFORMATION,  # 只改 DACL
                                                None,
                                                None,
                                                new_acl, # dacl 新名单
                                                None)
        if result != 0:
            raise OSError(f"SetNamedSecurityInfoW 失败：Win32 {result}")
        return True
    finally:
        if descriptor.value:
            kernel32.LocalFree(descriptor)
        if new_acl.value:
            kernel32.LocalFree(new_acl)


# ---------------------------------------------------------------------------
# 结果
# ---------------------------------------------------------------------------
@dataclass
class ConfinedRun:
    """一次受限运行的结果。字段与**历史上** docker 那一档的 `CommandResult` 对齐（多一个
    `enforcement`）—— 那是刻意的：接线的调用方换后端时不用改读数的地方。"""

    return_code: int | None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    duration_s: float = 0.0
    truncated: bool = False
    #: 隔离强度。这一档官方定性是 **partial**（`Everyone` 必须留在限制列表里、硬链接会别名化），
    #: 所以**不许**写成 "full" —— 那是自欺。
    enforcement: str = "partial"


# ---------------------------------------------------------------------------
# 令牌
# ---------------------------------------------------------------------------
def _logon_sid(token) -> ctypes.c_void_p:
    """从 `TokenGroups` 查询 `SE_GROUP_LOGON_ID` 的那个 SID，并对该 SID 内存数据做拷贝到新开辟的内存中，
    并返回指向这块内存的指针

    """
    advapi32 = _libs()["advapi32"]
    need = ctypes.c_uint32()
    # 给定令牌句柄，获取该句柄内的 SID集合组(不包括用户 SID)的字节数，该字节数可由 need 知道
    # 字节长度计数包含：包含 GroupCount ＋ 对齐填充 ＋ N 个 SID_AND_ATTRIBUTES ＋ 所有 SID 本身的数据
    advapi32.GetTokenInformation(token, TOKEN_GROUPS, None, 0, ctypes.byref(need))
    if not need.value:
        raise SandboxUnavailable(f"读 TokenGroups 长度失败：{_why()}")
    # 创建一个 need.value 字节的可变字节缓冲区，初始全是 0。
    buffer = ctypes.create_string_buffer(need.value)
    if not advapi32.GetTokenInformation(token, TOKEN_GROUPS, buffer, need.value, ctypes.byref(need)):
        raise SandboxUnavailable(f"读 TokenGroups 失败：{_why()}")
    groups = ctypes.cast(buffer, ctypes.POINTER(TokenGroups)).contents
    array = ctypes.cast(ctypes.byref(groups, TokenGroups.Groups.offset), ctypes.POINTER(SID_AND_ATTRIBUTES))
    for index in range(groups.GroupCount):
        entry = array[index]
        # 高位置 1 —— 用无符号比较，别被有符号数骗了
        # 该SID 是不是一个登录会话 SID
        if (entry.Attributes & 0xFFFFFFFF) & SE_GROUP_LOGON_ID:
            return _copy_sid(entry.Sid)
    raise SandboxUnavailable("令牌里找不到 logon SID")


def _everyone_sid() -> ctypes.c_void_p:
    """创建一个知名 SID，并返回指向这个 SID 的指针"""
    buffer = ctypes.create_string_buffer(68)
    size = ctypes.c_uint32(68)
    if not _libs()["advapi32"].CreateWellKnownSid(WIN_WORLD_SID, None, buffer, ctypes.byref(size)):
        raise SandboxUnavailable(f"CreateWellKnownSid(WinWorldSid) 失败：{_why()}")
    # 同上：`buffer` 是局部的，必须拷一份再交出去。
    return _copy_sid(ctypes.cast(buffer, ctypes.c_void_p))


def _merge_token_default_dacl(token, sid) -> None:
    """把一条拥有全权、类型为 允许 的 ACE 合并进令牌的**默认 DACL**。
    """
    advapi32, kernel32 = _libs()["advapi32"], _libs()["kernel32"]
    need = ctypes.c_uint32()
    # 获取令牌内容的默认dacl的占用字节数大小
    advapi32.GetTokenInformation(token, TOKEN_DEFAULT_DACL, None, 0, ctypes.byref(need))
    if not need.value:
        raise SandboxUnavailable(f"读 TokenDefaultDacl 长度失败：{_why()}")
    # 分配一块指定大小的内存空间
    buffer = ctypes.create_string_buffer(need.value)
    # 复制一份令牌内容上的默认dacl数据到新分配的那块内存上
    if not advapi32.GetTokenInformation(token, TOKEN_DEFAULT_DACL, buffer, need.value, ctypes.byref(need)):
        raise SandboxUnavailable(f"读 TokenDefaultDacl 失败：{_why()}")

    # 获取指向默认dacl副本的指针
    old_acl = ctypes.cast(buffer, ctypes.POINTER(TOKEN_DEFAULT_DACL_STRUCT)).contents.DefaultDacl

    # 构建一条指令：添加一条允许类型的ACE，该ACE拥有全权，该 entry 默认设置该ACE可被往下传
    entry = _explicit_access(sid, FILE_ALL_ACCESS, GRANT_ACCESS)
    new_acl = ctypes.c_void_p()

    # 往旧 dacl 添加一条新的 ace，得到新的 dacl
    result = advapi32.SetEntriesInAclW(1, ctypes.byref(entry), old_acl, ctypes.byref(new_acl))
    if result != 0:
        raise SandboxUnavailable(f"合并默认 DACL 失败（SetEntriesInAclW）：Win32 {result}")
    try:
        info = (ctypes.c_uint64 * 1)(new_acl.value or 0)
        # 往令牌内容写一份新的默认 dacl
        if not advapi32.SetTokenInformation(token, TOKEN_DEFAULT_DACL, info, ctypes.sizeof(info)):
            raise SandboxUnavailable(f"写 TokenDefaultDacl 失败：{_why()}")
    finally:
        if new_acl.value:
            kernel32.LocalFree(new_acl)


# ---------------------------------------------------------------------------
# 主体
# ---------------------------------------------------------------------------
class RestrictedTokenSandbox:
    """一个工作区上的宿主内沙箱（写受限令牌）。

    **生命周期**：``ensure()`` 挂授权 + 造令牌（幂等）；``run()`` 起一条受限命令；
    ``close()`` 按配置撤授权（幂等、**不抛异常** —— 它挂在收尾路径上）。

    **只限制写**：读不受限，所以解释器、venv、祖先目录、NUL 全都不需要额外授权 ——
    这是选它而不是 AppContainer 的全部理由。

    ## 临时目录：私有，不是共享的 `%TEMP%`

    `ensure()` 会在 `temp_root` 下建一个**本沙箱私有**的目录（`harness-sandbox-<6位十六进制>`），
    只给它挂授权，并把子进程的 `TEMP`/`TMP` **指到它**。这不是洁癖，是必须：

    - 授权共享的 `%TEMP%` 根 = 在墙上开了个洞。实测确认 DSH 也不是这么干的 —— 它建会话私有
      临时目录并把 `TEMP` 指过去（本会话现场：`TEMP=C:\\Users\\…\\Temp\\dsh-WqLVCi`），
      那条临时能力 SID 正是从**那个目录**的路径派生的。
    - 私有目录还顺带解决了"残留"问题：`close()` 撤完授权就把它整棵删掉，共享临时区里不留东西。

    代价：子进程的 `TEMP` 与父进程**不同**（这是有意的）。会话内跨命令的临时文件照旧能复用，
    因为那个目录在 `ensure()`→`close()` 之间一直活着。

    **不是线程安全的**（现在没有并发路径）。
    """

    def __init__(self, workspace: str, *, temp_root: str | None = None,
                 memory_mb: int | None = 2048, active_process_limit: int | None = 256,
                 revoke_on_close: bool = True, extra_env: Mapping[str, str] | None = None,
                 max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES) -> None:
        #: ★ `resolve()` 在**这里**做，而不是在算能力 SID 的时候 —— 种子必须是"定下形"的那个
        #: 字符串（见 `workspace_sid` 的说明：算种子时加规范化会让 SID 与 DSH 对不上）。
        self.workspace = str(Path(workspace).resolve())
        #: 私有临时目录的**父目录**（默认 `%TEMP%`）。
        self.temp_root = str(Path(temp_root or os.environ.get("TEMP")
                                  or os.environ.get("TMP") or self.workspace).resolve())
        #: 本沙箱私有的临时目录，`ensure()` 里建、`close()` 里删。
        self.temp_dir: str | None = None
        #: 追加/覆盖到子进程环境上的键值（`TEMP`/`TMP` 由本类自己管，别放进来）。
        #: 上层用它把 `PATH` 指到工作区的 venv —— 见 `bash.py`。
        self.extra_env: dict[str, str] = dict(extra_env or {})
        self.memory_mb = memory_mb
        self.active_process_limit = active_process_limit
        self.revoke_on_close = revoke_on_close
        self.max_output_bytes = max_output_bytes
        #: 能力 SID 的文本形式（工作区 / 临时目录各一个）。
        self.sids: dict[str, str] = {}
        self._token = ctypes.c_void_p()
        self._ensured = False
        #: 已经挂过授权的**根**（路径, SID, 掩码）—— 撤的时候照单撤，不去猜。
        #: 只有根：不递归，系统会自己把继承副本传播/移除（见 `_grant_all`）。
        self._granted: list[tuple[str, ctypes.c_void_p, int]] = []
        self.notes: list[str] = []

    # ---- 属性 ----
    @property
    def workspace_sid(self) -> str:
        if "workspace" not in self.sids:
            raise SandboxUnavailable("还没 ensure()")
        return self.sids["workspace"]

    # ---- 生命周期 ----
    def ensure(self) -> str:
        """挂授权 + 造受限令牌。幂等。返回工作区能力 SID 的文本形式。"""
        if self._ensured:
            return self.workspace_sid
        if not Path(self.workspace).is_dir():
            raise SandboxUnavailable(f"工作区不存在：{self.workspace}")

        self.sids = {"workspace": workspace_sid(self.workspace)}
        self.temp_dir = self._make_temp_dir()
        if self.temp_dir:
            self.sids["temp"] = temp_sid(self.temp_dir)

        try:
            self._grant_all()
            self._token = self._build_token()
        except Exception:
            # 半途失败 → 撤干净再原样抛（fail-closed，不留半套授权）。
            self._revoke_all()
            self._token = ctypes.c_void_p()
            # ★ 刚建出来的私有临时目录也要收掉。**顺序不能反**：先撤授权再删目录；
            # 反过来的话 `_revoke_all` 会去撤一个已经不存在的路径。这里失败是**要抛出去**的
            # （调用方得知道沙箱起不来），所以清理由 try 包住、吞掉自己的异常：
            # 清理没做成的坏消息不值得盖掉"为什么起不来"这个真消息。
            if self.temp_dir:
                try:
                    shutil.rmtree(self.temp_dir, ignore_errors=True)
                except Exception as exc:             # noqa: BLE001
                    self.notes.append(f"清理私有临时目录失败：{exc}")
                self.temp_dir = None
            raise
        self._ensured = True
        return self.workspace_sid

    def close(self) -> None:
        """按配置撤授权。**绝不抛异常**（它在会话收尾路径上）。"""
        if self.revoke_on_close:
            try:
                self._revoke_all()
            except Exception as exc:                 # noqa: BLE001 —— 收尾路径不许抛
                self.notes.append(f"撤授权失败：{exc}")
        if self._token.value:
            try:
                _libs()["kernel32"].CloseHandle(self._token)
            except Exception as exc:                 # noqa: BLE001
                self.notes.append(f"关令牌句柄失败：{exc}")
        self._token = ctypes.c_void_p()
        self._ensured = False

        if self.temp_dir:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            self.notes.append(f"已删私有临时目录：{self.temp_dir}")
            self.temp_dir = None

    # ---- 授权 ----
    def _make_temp_dir(self) -> str | None:
        """建本沙箱私有的临时目录。建不出来只记一笔、返回 `None`（不致命：沙箱照样能跑，
        只是子进程没有一个专属的可写临时区）。"""
        base = Path(self.temp_root)
        if not base.is_dir():
            self.notes.append(f"临时区父目录不存在，跳过私有临时目录：{base}")
            return None
        # 名字带随机 6 位十六进制：同一个父目录下可能同时活着多个沙箱（并发会话），
        # 拿固定名字会让第二个撞上"目录已存在"。
        for _ in range(8):
            candidate = base / f"harness-sandbox-{os.urandom(3).hex()}"
            try:
                candidate.mkdir()
            except FileExistsError:
                continue
            except OSError as exc:
                self.notes.append(f"建私有临时目录失败：{exc}")
                return None
            self.notes.append(f"私有临时目录：{candidate}")
            return str(candidate)
        self.notes.append("私有临时目录连撞 8 次名字，放弃（这不正常）")
        return None

    def _grant_all(self) -> None:
        """给**两个根**各挂一次授权
        """
        # 工作区：可写可删。ACE 带 `OI|CI`，系统会把它传播到已存在的后代，将来新建的自动继承。
        self._grant(self.workspace, self.sids["workspace"])
        # 私有临时目录：目录是刚建的、里面是空的，只需要根这一条（pytest 要在这里建
        # `pytest-of-*/` 与临时文件，靠继承位就够了）。
        if self.temp_dir and "temp" in self.sids:
            self._grant(self.temp_dir, self.sids["temp"])

    def _grant(self, path: str, sid_text: str) -> None:
        """给**一个对象**（工作区根 / 私有临时目录根）挂授权。返回是否真的改了 ACL。
        """

        sid = _to_sid(sid_text)
        try:
            # 向目标可安全对象的安全描述符的 DACE 添加一条 类型为 【运行】的、拥有修改类权限的 ACE
            changed = _apply_ace(path, sid, MASK_MODIFY, GRANT_ACCESS, skip_if_present=True)    # MASK_MODIFY 是修改类权限掩码，包括写、追加、改属性、删
        except OSError as exc:
            _libs()["kernel32"].LocalFree(sid)
            raise SandboxUnavailable(f"给 {path} 挂授权失败：{exc}") from exc
        self._granted.append((path, sid, MASK_MODIFY))
        label = "工作区" if path == self.workspace else "私有临时目录"
        self.notes.append(f"{label} {path}（SID {sid_text}）："
                          f"{'新增 1（系统会自行传播到已有后代）' if changed else '已存在（跳过）'}")

    def _revoke_all(self) -> None:
        """撤销掉指定路径的文件系统对象的指定 SID 的 DACL 中的 ACE"""
        kernel32 = _libs()["kernel32"]
        for path, sid, mask in reversed(self._granted):
            try:
                _apply_ace(path, sid, mask, REVOKE_ACCESS)
            except OSError as exc:
                self.notes.append(f"撤 {path} 的授权失败：{exc}")
            finally:
                kernel32.LocalFree(sid)
        self._granted.clear()

    # ---- 令牌 ----
    def _build_token(self) -> ctypes.c_void_p:
        """获取当前进程的伪句柄，使用伪句柄获取该进程的访问令牌(并拥有四种令牌操作权限)的句柄，
        接着构建一份 SID 限制列表， 使用访问令牌的句柄和限制列表创建一个写受限的主令牌，
        最后往这个新的主令牌内容的默认 DACL 加入一个拥有全权、类型为 【允许】可被往下传的 ACE，最后返回这个新的主令牌"""
        advapi32, kernel32 = _libs()["advapi32"], _libs()["kernel32"]
        raw = ctypes.c_void_p()

        # kernel32.GetCurrentProcess()：获取当前进程的伪句柄
        # 获取当前进程的伪句柄，这个句柄用于获取当前进程的令牌句柄，
        # 并申请【拿它当新进程的主令牌、复制这个令牌、读令牌里的内容和改令牌的默认DACL】这四组操作，并将令牌句柄保存到raw.value中
        if not advapi32.OpenProcessToken(kernel32.GetCurrentProcess(), TOKEN_OPEN_MASK, ctypes.byref(raw)):
            raise SandboxUnavailable(f"OpenProcessToken 失败：{_why()}"
                "（掩码里必须带 TOKEN_ASSIGN_PRIMARY，否则后面 CreateProcessAsUserW 报 5）")
        #: 本函数造出来的 SID，收尾一次性释放（都是 LocalAlloc 家族）。
        owned: list[ctypes.c_void_p] = []
        try:
            owned.append(_logon_sid(raw))
            owned.append(_everyone_sid())
            owned.extend(_to_sid(text) for text in self.sids.values())
            # 限制列表
            restricting = owned
            # 是创建一个PyCArrayType实例，数组元素个数是len(restricting)个，
            # 然后为登录会话SID、知名SID以及self.sids.values()构成的 SID 指针，
            # 分别创建一个SID_AND_ATTRIBUTES实例，其字段Sid就是前边提到的各个Sid 指针，
            # 但是所有SID_AND_ATTRIBUTES的Attributes字段的值都是0，也只能是0，否则会报错
            array = (SID_AND_ATTRIBUTES * len(restricting))(*[SID_AND_ATTRIBUTES(sid, 0) for sid in restricting])
            token = ctypes.c_void_p()

            # 创建一个新的主令牌，是一个写受限令牌
            # 创建新令牌时，新令牌的安全描述符是从模板令牌照抄下来的。
            if not advapi32.CreateRestrictedToken(raw,              # 拿哪张令牌当模板
                                                  RESTRICTED_TOKEN_FLAGS, # 三个开关
                                                  0, None,                # 从raw继承的组列表，不禁用任何组
                                                  0, None,                # 特权一个都不列
                                                  len(restricting), array, # 要【加进限制栏】的 SID
                                                  ctypes.byref(token)):
                raise SandboxUnavailable(f"CreateRestrictedToken 失败：{_why()}")

            try:
                _merge_token_default_dacl(token, restricting[-1])
            except SandboxUnavailable as exc:
                self.notes.append(f"合并令牌默认 DACL 失败（继续；孙进程自己开管道可能受影响）：{exc}")
            return token
        finally:
            # 关闭令牌句柄
            kernel32.CloseHandle(raw)

            # 释放每一个 SID 指针指向的内存
            for sid in owned:
                kernel32.LocalFree(sid)

    # ---- 起进程 ----
    def run(self, argv: list[str], *, cwd: str | None = None, timeout_s: float = 60.0) -> ConfinedRun:
        """起一条**受限**命令。**超时也返回已经读到的输出**（工具契约的硬要求）。"""
        self.ensure()
        return self._launch(list(argv), cwd or self.workspace, timeout_s)

    def _launch(self, argv: list[str], cwd: str, timeout_s: float) -> ConfinedRun:
        kernel32 = _libs()["kernel32"]
        started = time.monotonic()

        out_r, out_w = self._pipe()
        err_r, err_w = self._pipe()
        nul = self._open_nul()

        out_stream = self._as_stream(out_r)
        err_stream = self._as_stream(err_r)
        captured: dict[str, tuple[bytes, bool]] = {}

        # 从输出管道和异常管道的读端获取管道内的数据
        readers = [
            threading.Thread(target=self._drain, args=(key, stream, captured), daemon=True)
            for key, stream in (("stdout", out_stream), ("stderr", err_stream))
        ]
        for reader in readers:
            reader.start()

        job = 0
        process = 0
        thread = 0
        timed_out = False
        try:
            job = self._create_job()

            process, thread = self._create_process(argv, cwd, [nul, out_w, err_w])
            # 父进程必须放掉写端，否则读者永远等不到 EOF。
            # **关掉之后要清零**：否则下面的 finally 会再关一次同一个句柄号，
            # 而那个号可能已经被内核复用给别人了。
            # 这里是给主进程关闭输出/异常管道的写端，之后主进程只持有这两个管道的读端句柄，而主进程的子进程正好相反，只持有这两个管道的写端句柄
            for holder in (out_w, err_w):
                kernel32.CloseHandle(holder)
            out_w = err_w = 0

            # 把 process 这个进程，加进 job 这个 Job Object 里。
            # 加进去之后能：
            #   对 Job 调 TerminateJobObject → 里面的进程全被杀掉
            #   给 Job 设的限制（内存上限、进程数上限）→ 对里面的进程生效
            #   "后代也自动在里面"是 Windows 的规矩：一个进程在某个 Job 里，它以后创建的进程自动也在同一个 Job 里。
            # 父进程只要管住一个 Job，就能管住子进程、以及子进程再创建的进程。
            if not kernel32.AssignProcessToJobObject(job, process):
                reason = _why()
                # 这个失败最容易被误判成"沙箱坏了"。先把最可能的原因点出来（见 `_in_a_job`）。
                # **补救方向**（没有实测复现之前不写进去，免得留一段永远跑不到的代码）：
                # 给 `CreateProcessAsUserW` 加 `CREATE_BREAKAWAY_FROM_JOB` 重试。
                hint = ("（本进程自己就在一个 Job 里 —— 若父 Job 不允许嵌套，这就是原因）" if _in_a_job() else "")
                kernel32.TerminateProcess(process, 1)
                raise SandboxUnavailable(f"放进 Job Object 失败：{reason}{hint}")

            # 将进程的主线程从挂起态转变成运行态
            if kernel32.ResumeThread(thread) == 0xFFFFFFFF:
                raise SandboxUnavailable(f"ResumeThread 失败：{_why()}")
            kernel32.CloseHandle(thread)
            thread = 0
            # 等待，并校验是不是超时了
            if kernel32.WaitForSingleObject(process, int(timeout_s * 1000)) == WAIT_TIMEOUT:
                timed_out = True
                # 杀**整棵树**：只管子进程的话，它拉起的孙进程会攥着管道不放，读者等不到 EOF
                kernel32.TerminateJobObject(job, 1)
                kernel32.WaitForSingleObject(process, 5000)
            code = ctypes.c_uint32()

            # 获取 process 的退出码
            kernel32.GetExitCodeProcess(process, ctypes.byref(code))
            return_code = None if timed_out else int(code.value)
        finally:
            # 逐个关闭所有句柄
            for holder in (nul, out_w, err_w, thread, process):
                if holder:
                    kernel32.CloseHandle(holder)
            if job:
                # 关掉最后一个 job 句柄 = KILL_ON_JOB_CLOSE 生效，孤儿进程在这里被收掉
                kernel32.CloseHandle(job)

        for reader in readers:
            # 如果线程已经结束则立即返回，否则最多等5s
            reader.join(timeout=5.0)
        out, out_cut = captured.get("stdout", (b"", False))
        err, err_cut = captured.get("stderr", (b"", False))
        return ConfinedRun(return_code=return_code,
                           stdout=_decode_output(out), stderr=_decode_output(err),
                           timed_out=timed_out, duration_s=time.monotonic() - started,
                           truncated=out_cut or err_cut)

    # ---- 起进程用的小件 ----
    def _pipe(self) -> tuple[int, int]:
        """创建一根可继承的管道，并让读端不可继承（不给子进程），最后返回这个管道的读/写端句柄"""
        kernel32 = _libs()["kernel32"]
        sa = SECURITY_ATTRIBUTES(ctypes.sizeof(SECURITY_ATTRIBUTES), None, True)
        read_end, write_end = ctypes.c_void_p(), ctypes.c_void_p()

        # 造一根管道，并持有管道两端的句柄
        if not kernel32.CreatePipe(ctypes.byref(read_end), # 出参：把读端句柄写到这儿
                                   ctypes.byref(write_end), # 出参：把写端句柄写到这儿
                                   ctypes.byref(sa),    #表单
                                   0):      # 期待管子的缓冲区大小；0 = 用系统默认
            raise SandboxUnavailable(f"CreatePipe 失败：{_why()}")

        # 把读端句柄改成不可被子进程继承。
        if not kernel32.SetHandleInformation(read_end, # 修改的是读端句柄
                                             HANDLE_FLAG_INHERIT, # HANDLE_FLAG_INHERIT指的是是否可以被子进程继承
                                             0):    # 对第二个参数对应的开关的设置，0不可以，1可以
            raise SandboxUnavailable(f"SetHandleInformation 失败：{_why()}")
        return _h(read_end), _h(write_end)

    def _open_nul(self) -> int:
        """打开一个NUL设备，并申请这个设备的读权限，允许别人同时读/写这个设备，这个设备是已存在不用新建；
        然后得到这次打开这个设备的句柄(不是设备的句柄)，这个句柄能被赋值给子进程，
        把句柄值统一成一个纯 int，并约定 0 = 没有句柄， 然后把句柄值返回"""
        sa = SECURITY_ATTRIBUTES(ctypes.sizeof(SECURITY_ATTRIBUTES), None, True)


        handle = _libs()["kernel32"].CreateFileW("NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE,
            ctypes.byref(sa), OPEN_EXISTING, 0, None)
        if not handle:
            raise SandboxUnavailable(f"打开 NUL 失败：{_why()}")
        return _h(handle)

    @staticmethod
    def _as_stream(handle: int):
        """把句柄登记到“打开文件表”，取得句柄在表中的文件描述符，用这个文件描述符包装成一个 python 文件对象， 并返回这个文件对象"""
        import msvcrt                       # Windows 专有：模块顶层 import 会在 Linux 上炸
        # 把这个句柄登记进 C 运行库的"打开文件表"，然后返回它在这张表里的编号。那个编号就是 fd 文件描述符，这个文件描述符只读、传输模式是二进制数据传输
        fd = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
        if fd < 0:
            raise SandboxUnavailable("open_osfhandle 失败（读端转不成文件对象）")
        # 把这个 fd 包成一个 Python 文件对象，返回出去。
        return os.fdopen(fd, "rb", buffering=0)

    def _drain(self, key: str, stream, sink: dict) -> None:
        """读干一根管道，只留前 `max_output_bytes` 字节。

        **超了也要继续读到 EOF**：不排空的话，写满内核缓冲区的子进程会卡在 write 上不往前
        走，我们的等待就永远等不到它 —— "限制输出"于是变成"制造死锁"。
        """
        kept: list[bytes] = []
        size = 0
        cut = False
        while True:
            try:
                block = stream.read(_READ_CHUNK)
            except OSError:
                break
            if not block:
                break

            # 累积得到的数据字节流长度到达self.max_output_bytes之后，后续的数据字节流就不再追加到kept中
            # 但还不能 break，需要将管道内的数据排干，只是不追加到kept中
            room = self.max_output_bytes - size
            if room <= 0:
                cut = True
                continue

            # 如果数据字节流长度大于剩余空间长度，只取block前N个字节，N指的是剩余字节长度
            if len(block) > room:
                cut = True
                block = block[:room]
            kept.append(block)  # 数据字节流列表
            size += len(block)  # 数据流累计字节数

        # 将各个数据字节流按顺序拼接到一起
        sink[key] = (b"".join(kept), cut)
        try:
            stream.close()
        except OSError:
            pass

    def _create_job(self) -> int:
        """创建一个Job Object（作业对象），并给它装三条限制：
            关掉 Job 的最后一个句柄时，Job 里所有进程全被杀掉；

            每个进程的内存上限（默认 2048 MB）；

            最多同时几个进程（默认 256）
        最后返回这个 Job Object 的句柄值"""
        kernel32 = _libs()["kernel32"]
        job = kernel32.CreateJobObjectW(None, None)

        if not job:
            raise SandboxUnavailable(f"CreateJobObjectW 失败：{_why()}")

        limits = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE

        if self.memory_mb:
            limits.BasicLimitInformation.LimitFlags |= JOB_OBJECT_LIMIT_PROCESS_MEMORY
            limits.ProcessMemoryLimit = int(self.memory_mb) * 1024 * 1024

        if self.active_process_limit:
            limits.BasicLimitInformation.LimitFlags |= JOB_OBJECT_LIMIT_ACTIVE_PROCESS
            limits.BasicLimitInformation.ActiveProcessLimit = int(self.active_process_limit)

        if not kernel32.SetInformationJobObject(
                job, JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS,
                ctypes.byref(limits), ctypes.sizeof(limits)):
            kernel32.CloseHandle(job)
            raise SandboxUnavailable(f"SetInformationJobObject 失败：{_why()}")
        return _h(job)

    def _environment_block(self) -> str | None:
        """获取父进程的环境变量，并造子进程的环境块：`TEMP`/`TMP` 指到**私有临时目录**，再加上 `extra_env`。
        """
        if not self.temp_dir and not self.extra_env:
            return None
        env = {key: value for key, value in os.environ.items() if key.upper() not in ("TEMP", "TMP")}
        if self.temp_dir:
            env["TEMP"] = self.temp_dir
            env["TMP"] = self.temp_dir
        if self.extra_env:
            overridden = {key.upper() for key in self.extra_env}
            env = {key: value for key, value in env.items() if key.upper() not in overridden}
            env.update(self.extra_env)
        return "".join(f"{key}={value}\0" for key, value in env.items()) + "\0"

    def _create_process(self, argv: list[str], cwd: str, inherited: list[int]) -> tuple[int, int]:
        """使用新建的受限令牌创建一个新的进程（挂起主线程），然后往该进程传递执行指令，并返回（进程句柄, 线程句柄），并未真的去执行指令。

        argv: 指令列表

        cwd: 当前工作路径

        inherited: 句柄列表，依次是： NUL 设备，输出写端、异常写端
        """
        advapi32 = _libs()["advapi32"]
        si = STARTUPINFOW()
        si.cb = ctypes.sizeof(STARTUPINFOW)
        si.dwFlags = STARTF_USESTDHANDLES   # 标识不要使用系统的三个标准句柄而是使用 inherited 内的三个句柄
        si.hStdInput = inherited[0]
        si.hStdOutput = inherited[1]
        si.hStdError = inherited[2]
        pi = PROCESS_INFORMATION()
        env_block = self._environment_block()
        #: ★ 标志位必须跟着环境块走：给了 Unicode 环境块却没这个标志，中文路径会变乱码，
        #: 而报错只会表现为"文件和目录都找不到"。
        flags = CREATE_SUSPENDED    # 创建仅进程之后，先把主线程挂起
        if env_block is not None:
            flags |= CREATE_UNICODE_ENVIRONMENT # 按 UTF-16 编码模式读取内存中的数据

        # subprocess.list2cmdline:把 Python 的参数列表拼成一条命令行，并处理引号和转义,
        # ctypes.create_unicode_buffer: 把那个字符串变成一块可写的 UTF-16 内存
        command_line = ctypes.create_unicode_buffer(subprocess.list2cmdline(argv))

        if not advapi32.CreateProcessAsUserW(self._token, argv[0], command_line, None, None, True,
                flags, env_block, cwd, ctypes.byref(si), ctypes.byref(pi)):
            raise SandboxUnavailable(f"CreateProcessAsUserW 失败：{_why()}.（argv={argv!r} cwd={cwd!r}）")
        return _h(pi.hProcess), _h(pi.hThread)


def available() -> str | None:
    """环境能不能用这一档：可用返回 `None`，否则返回原因（给调用方决定要不要 fail-closed）。"""
    if sys.platform != "win32":
        return f"只在 Windows 上可用（当前 {sys.platform}）"
    try:
        _libs()
    except SandboxUnavailable as exc:
        return str(exc)
    return None