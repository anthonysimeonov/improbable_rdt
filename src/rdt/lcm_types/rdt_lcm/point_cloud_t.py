"""LCM type definitions
This file automatically generated by lcm.
DO NOT MODIFY BY HAND!!!!
"""

try:
    import cStringIO.StringIO as BytesIO
except ImportError:
    from io import BytesIO
import struct

import rdt_lcm.header_t

class point_cloud_t(object):
    __slots__ = ["header", "num_points", "data_size", "data"]

    __typenames__ = ["rdt_lcm.header_t", "int32_t", "int64_t", "byte"]

    __dimensions__ = [None, None, None, ["data_size"]]

    def __init__(self):
        self.header = rdt_lcm.header_t()
        self.num_points = 0
        self.data_size = 0
        self.data = b""

    def encode(self):
        buf = BytesIO()
        buf.write(point_cloud_t._get_packed_fingerprint())
        self._encode_one(buf)
        return buf.getvalue()

    def _encode_one(self, buf):
        assert self.header._get_packed_fingerprint() == rdt_lcm.header_t._get_packed_fingerprint()
        self.header._encode_one(buf)
        buf.write(struct.pack(">iq", self.num_points, self.data_size))
        buf.write(bytearray(self.data[:self.data_size]))

    def decode(data):
        if hasattr(data, 'read'):
            buf = data
        else:
            buf = BytesIO(data)
        if buf.read(8) != point_cloud_t._get_packed_fingerprint():
            raise ValueError("Decode error")
        return point_cloud_t._decode_one(buf)
    decode = staticmethod(decode)

    def _decode_one(buf):
        self = point_cloud_t()
        self.header = rdt_lcm.header_t._decode_one(buf)
        self.num_points, self.data_size = struct.unpack(">iq", buf.read(12))
        self.data = buf.read(self.data_size)
        return self
    _decode_one = staticmethod(_decode_one)

    def _get_hash_recursive(parents):
        if point_cloud_t in parents: return 0
        newparents = parents + [point_cloud_t]
        tmphash = (0x916a6679ddd821ad+ rdt_lcm.header_t._get_hash_recursive(newparents)) & 0xffffffffffffffff
        tmphash  = (((tmphash<<1)&0xffffffffffffffff) + (tmphash>>63)) & 0xffffffffffffffff
        return tmphash
    _get_hash_recursive = staticmethod(_get_hash_recursive)
    _packed_fingerprint = None

    def _get_packed_fingerprint():
        if point_cloud_t._packed_fingerprint is None:
            point_cloud_t._packed_fingerprint = struct.pack(">Q", point_cloud_t._get_hash_recursive([]))
        return point_cloud_t._packed_fingerprint
    _get_packed_fingerprint = staticmethod(_get_packed_fingerprint)

    def get_hash(self):
        """Get the LCM hash of the struct"""
        return struct.unpack(">Q", point_cloud_t._get_packed_fingerprint())[0]

