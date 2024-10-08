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

import rdt_lcm.pose_t

class pose_stamped_t(object):
    __slots__ = ["header", "pose"]

    __typenames__ = ["rdt_lcm.header_t", "rdt_lcm.pose_t"]

    __dimensions__ = [None, None]

    def __init__(self):
        self.header = rdt_lcm.header_t()
        self.pose = rdt_lcm.pose_t()

    def encode(self):
        buf = BytesIO()
        buf.write(pose_stamped_t._get_packed_fingerprint())
        self._encode_one(buf)
        return buf.getvalue()

    def _encode_one(self, buf):
        assert self.header._get_packed_fingerprint() == rdt_lcm.header_t._get_packed_fingerprint()
        self.header._encode_one(buf)
        assert self.pose._get_packed_fingerprint() == rdt_lcm.pose_t._get_packed_fingerprint()
        self.pose._encode_one(buf)

    def decode(data):
        if hasattr(data, 'read'):
            buf = data
        else:
            buf = BytesIO(data)
        if buf.read(8) != pose_stamped_t._get_packed_fingerprint():
            raise ValueError("Decode error")
        return pose_stamped_t._decode_one(buf)
    decode = staticmethod(decode)

    def _decode_one(buf):
        self = pose_stamped_t()
        self.header = rdt_lcm.header_t._decode_one(buf)
        self.pose = rdt_lcm.pose_t._decode_one(buf)
        return self
    _decode_one = staticmethod(_decode_one)

    _hash = None
    def _get_hash_recursive(parents):
        if pose_stamped_t in parents: return 0
        newparents = parents + [pose_stamped_t]
        tmphash = (0xe10feebec5c97663+ rdt_lcm.header_t._get_hash_recursive(newparents)+ rdt_lcm.pose_t._get_hash_recursive(newparents)) & 0xffffffffffffffff
        tmphash  = (((tmphash<<1)&0xffffffffffffffff) + (tmphash>>63)) & 0xffffffffffffffff
        return tmphash
    _get_hash_recursive = staticmethod(_get_hash_recursive)
    _packed_fingerprint = None

    def _get_packed_fingerprint():
        if pose_stamped_t._packed_fingerprint is None:
            pose_stamped_t._packed_fingerprint = struct.pack(">Q", pose_stamped_t._get_hash_recursive([]))
        return pose_stamped_t._packed_fingerprint
    _get_packed_fingerprint = staticmethod(_get_packed_fingerprint)

