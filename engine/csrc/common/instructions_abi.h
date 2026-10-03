#pragma once

// Generated ABI mirror for vinf runtime/instructions.py.
#define VINF_INTS_PER_INSTRUCTION 32

#define VINF_OPCODE_NOOP 0
#define VINF_OPCODE_RMS_QKV_ROPE 1
#define VINF_OPCODE_ATTENTION 2
#define VINF_OPCODE_O_PROJ 3
#define VINF_OPCODE_MLP_UPGATE 4
#define VINF_OPCODE_DOWN_PROJ 5
#define VINF_OPCODE_LM_HEAD 6
#define VINF_OPCODE_VERIFY 7

// Word 0 is always opcode.
// RMSQKVRope: [1]=layer_idx [2]=start_block [3]=end_block [4]=position
// Attention:  [1]=layer_idx [2]=kv_head_idx [3]=start_position [4]=end_position
// OProj:      [1]=layer_idx [2]=start_block [3]=end_block
// MLPUpGate:  [1]=layer_idx [2]=start_block [3]=end_block
// DownProj:   [1]=layer_idx [2]=start_block [3]=end_block
// LMHead:     [1]=start_vocab_block [2]=end_vocab_block
// Verify:     [1]=position [2]=num_verify_tokens [3]=draft_token_start
