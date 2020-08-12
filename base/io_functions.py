import io
import struct


def throw_on_error(ok, info=""):
    if not ok:
        raise RuntimeError(info)


def peek_char(fd, num_chars=1):
    ch = fd.peek(num_chars)[:num_chars]
    return bytes.decode(ch)


def expect_binary(fd):
    flags = bytes.decode(fd.read(2))
    throw_on_error(flags == '\0B', f"Expect binary flag, but gets {flags}")


def read_token(fd):
    key = ""
    while True:
        c = bytes.decode(fd.read(1))
        if c == ' ' or c == '':
            break
        key += c
    return None if key == '' else key.strip()


def write_token(fd, token):
    fd.write(str.encode(token + " "))


def write_binary_symbol(fd):
    fd.write(str.encode('\0B'))


def read_int32(fd):
    int_size = bytes.decode(fd.read(1))
    throw_on_error(int_size == '\04', f"Expert '\\04', but gets {int_size}")
    int_str = fd.read(4)
    int_val = struct.unpack("i", int_str)
    return int_val[0]


def write_int32(fd, int32):
    fd.write(str.encode('\04'))
    int_pack = struct.pack("i", int32)
    fd.write(int_pack)


def read_float32(fd):
    float_size = bytes.decode(fd.read(1))
    throw_on_error(float_size == '\04', f"Expect '\\04', but gets {float_size}")
    float_str = fd.read(4)
    float_val = struct.unpack('f', float_str)
    return float_val
