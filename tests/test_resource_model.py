import unittest

from model.resource_model import (
    A100_CC80_64K_SHARED,
    KernelResources,
    SmLimits,
    model_occupancy,
)


class ResourceModelTests(unittest.TestCase):
    def test_naive_profile_limits_are_reproduced(self) -> None:
        result = model_occupancy(KernelResources(256, 32, 0))

        self.assertEqual(result.block_limit_sm, 32)
        self.assertEqual(result.block_limit_threads, 8)
        self.assertEqual(result.block_limit_warps, 8)
        self.assertEqual(result.block_limit_registers, 8)
        self.assertEqual(result.block_limit_shared_memory, 64)
        self.assertEqual(result.resident_blocks, 8)
        self.assertEqual(result.resident_warps, 64)
        self.assertEqual(result.theoretical_occupancy, 1.0)
        self.assertEqual(
            result.limiting_resources, ("threads", "warps", "registers")
        )

    def test_tiled_profile_shared_memory_limit_is_reproduced(self) -> None:
        result = model_occupancy(KernelResources(256, 32, 2 * 16 * 16 * 4))

        self.assertEqual(result.allocated_shared_memory_per_block, 3 * 1024)
        self.assertEqual(result.block_limit_shared_memory, 21)
        self.assertEqual(result.resident_blocks, 8)
        self.assertEqual(result.theoretical_occupancy, 1.0)

    def test_register_allocation_is_rounded_per_warp(self) -> None:
        result = model_occupancy(KernelResources(64, 33, 0))

        self.assertEqual(result.allocated_registers_per_block, 2 * 1280)
        self.assertEqual(result.block_limit_registers, 25)

    def test_custom_device_can_be_modeled(self) -> None:
        device = SmLimits("test", 64, 2, 4, 1024, 4096)
        result = model_occupancy(KernelResources(32, 8, 0), device)

        self.assertEqual(result.resident_blocks, 2)
        self.assertEqual(result.theoretical_occupancy, 1.0)

    def test_invalid_resources_are_rejected(self) -> None:
        for resources in (
            (0, 32, 0),
            (256, 0, 0),
            (256, 32, -1),
        ):
            with self.subTest(resources=resources):
                with self.assertRaises(ValueError):
                    KernelResources(*resources)

    def test_a100_limits_are_named_explicitly(self) -> None:
        self.assertEqual(A100_CC80_64K_SHARED.max_threads, 2048)
        self.assertEqual(A100_CC80_64K_SHARED.max_warps, 64)
        self.assertEqual(A100_CC80_64K_SHARED.max_blocks, 32)
        self.assertEqual(A100_CC80_64K_SHARED.registers, 65_536)


if __name__ == "__main__":
    unittest.main()
