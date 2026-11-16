import unittest

import torch

from storage.state_quantization import (
    CODEC_FP16,
    CODEC_INT8,
    CODEC_FP32,
    StatePayload,
    decode_state,
    encode_state,
    get_codebook,
    validate_state_payload,
    HEADER,
)


class StateQuantizationTest(unittest.TestCase):
    def test_dynamic_codebooks_are_deterministic_and_have_expected_ranges(self):
        signed_a = get_codebook(0)
        signed_b = get_codebook(0)
        positive = get_codebook(1)
        self.assertTrue(torch.equal(signed_a, signed_b))
        self.assertLess(float(signed_a.min()), 0.0)
        self.assertGreater(float(signed_a.max()), 0.0)
        self.assertGreaterEqual(float(positive.min()), 0.0)
        self.assertGreater(float(positive.max()), 0.0)

    def test_int8_round_trip_preserves_shape_and_exact_zeros(self):
        state = torch.zeros((7, 118), dtype=torch.float32)
        state[:, 0] = torch.linspace(-0.25, 0.25, 7)
        state[:, 59] = torch.tensor([0.0, 1.0e-5, 2.0e-4, 1.0e-3, 1.0e-2, 0.1, 0.5])
        payload = encode_state(state, quantization="int8_dynamic", qtile_size=8)
        self.assertEqual(payload.codec, CODEC_INT8)
        validate_state_payload(payload)
        restored = decode_state(payload)
        self.assertEqual(tuple(restored.shape), tuple(state.shape))
        self.assertTrue(torch.equal(restored[state == 0], torch.zeros_like(restored[state == 0])))
        self.assertTrue(torch.isfinite(restored).all())

    def test_component_group_scatter_restores_all_columns(self):
        state = torch.arange(3 * 118, dtype=torch.float32).reshape(3, 118) / 17.0
        payload = encode_state(state, quantization="int8_dynamic", qtile_size=7)
        restored = decode_state(payload)
        self.assertEqual(tuple(restored.shape), tuple(state.shape))
        self.assertTrue(torch.isfinite(restored).all())
        self.assertTrue(torch.allclose(restored, state, atol=0.2, rtol=0.2))

    def test_fp16_and_fp32_payloads(self):
        state = torch.randn((3, 118), dtype=torch.float32)
        fp16 = encode_state(state, quantization="fp16")
        fp32 = encode_state(state, quantization="off")
        self.assertEqual(fp16.codec, CODEC_FP16)
        self.assertEqual(fp32.codec, CODEC_FP32)
        self.assertEqual(tuple(decode_state(fp16).shape), tuple(state.shape))
        self.assertTrue(torch.equal(decode_state(fp32), state))

    def test_corrupt_payload_is_rejected(self):
        state = torch.ones((2, 118), dtype=torch.float32)
        payload = encode_state(state, quantization="int8_dynamic", qtile_size=16)
        corrupt = StatePayload(
            payload.codec,
            payload.row_count,
            payload.raw_size,
            payload.payload[:-1],
            payload.qtile_size,
            payload.codebook_id,
        )
        with self.assertRaises(ValueError):
            validate_state_payload(corrupt)

    def test_unknown_codebook_is_rejected(self):
        state = torch.ones((2, 118), dtype=torch.float32)
        payload = encode_state(state, quantization="int8_dynamic", qtile_size=16)
        data = bytearray(payload.payload)
        header = list(HEADER.unpack_from(data, 0))
        header[10] = 99
        data[: HEADER.size] = HEADER.pack(*header)
        corrupt = StatePayload(
            payload.codec,
            payload.row_count,
            payload.raw_size,
            bytes(data),
            payload.qtile_size,
            payload.codebook_id,
        )
        with self.assertRaises(ValueError):
            validate_state_payload(corrupt)


if __name__ == "__main__":
    unittest.main()
