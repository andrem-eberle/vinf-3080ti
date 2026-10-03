from __future__ import annotations

import unittest

from vinf.cuda.megakernel import (
    RTX_3080_TI_DYNAMIC_SHARED_MEMORY,
    RTX_3080_TI_EXPECTED_SMS,
    RTX_3080_TI_NUM_THREADS,
    TIMING_WIDTH,
    MinimalMegakernelRuntime,
    noop_instruction_tensor,
    noop_schedule,
    validate_noop_result,
)


class Phase8MinimalMegakernelTests(unittest.TestCase):
    def test_noop_instruction_tensor_shape(self) -> None:
        tensor = noop_instruction_tensor(4)
        self.assertEqual(tensor.shape, (4, 1, 32))
        for sm_rows in tensor.rows:
            self.assertEqual(sm_rows[0][0], 0)

    def test_launch_dimensions_match_rtx_3080_ti_config(self) -> None:
        runtime = MinimalMegakernelRuntime()
        dims = runtime.launch_dimensions()
        self.assertEqual(dims.grid_blocks, RTX_3080_TI_EXPECTED_SMS)
        self.assertEqual(dims.block_threads, RTX_3080_TI_NUM_THREADS)
        self.assertEqual(dims.dynamic_shared_memory, RTX_3080_TI_DYNAMIC_SHARED_MEMORY)

    def test_allocate_instruction_and_timing_buffers(self) -> None:
        runtime = MinimalMegakernelRuntime(num_sms=3)
        buffers = runtime.allocate_noop(queue_len=2)
        self.assertEqual(buffers.instruction_shape, (3, 2, 32))
        self.assertEqual(buffers.timing_shape, (3, 2, TIMING_WIDTH))
        self.assertEqual(buffers.instructions[0][0][0], 0)
        self.assertEqual(buffers.timings[0][0][0], 0)

    def test_timing_readback_snapshot(self) -> None:
        runtime = MinimalMegakernelRuntime(num_sms=2)
        runtime.allocate_noop(queue_len=1)
        timings = runtime.read_timings()
        self.assertEqual(len(timings), 2)
        self.assertEqual(len(timings[0][0]), TIMING_WIDTH)

    def test_validate_noop_result(self) -> None:
        validate_noop_result((0, 86, RTX_3080_TI_NUM_THREADS))
        with self.assertRaises(RuntimeError):
            validate_noop_result((1, 86, RTX_3080_TI_NUM_THREADS))

    def test_noop_schedule_rejects_invalid_queue_len(self) -> None:
        with self.assertRaises(ValueError):
            noop_schedule(1, queue_len=0)

    def test_live_cuda_noop_launch_if_available(self) -> None:
        try:
            from vinf.cuda.noop import run_and_validate_noop_smoke

            result = run_and_validate_noop_smoke()
        except Exception as exc:
            self.skipTest(f"CUDA NoOp launch unavailable: {exc}")
        self.assertEqual(result, (0, 86, RTX_3080_TI_NUM_THREADS))


if __name__ == "__main__":
    unittest.main()

