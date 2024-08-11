/*
 * SPDX-FileCopyrightText: Copyright (c) 2003-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the Software),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED AS IS, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef __lr10_dev_nv_xp_h__
#define __lr10_dev_nv_xp_h__
/* This file is autogenerated.  Do not edit */
#define NV_XP_LANE_ERROR_STATUS                                     0x0008D400 /* RW-4R */
#define NV_XP_LANE_ERROR_STATUS_SYNC_HDR_CODING_ERR                    0:0 /* RWIVF */
#define NV_XP_LANE_ERROR_STATUS_SYNC_HDR_CODING_ERR_NOT_ACTIVE  0x00000000 /* R-I-V */
#define NV_XP_LANE_ERROR_STATUS_SYNC_HDR_CODING_ERR_ACTIVE      0x00000001 /* R---V */
#define NV_XP_LANE_ERROR_STATUS_SYNC_HDR_ORDER_ERR                     1:1 /* RWIVF */
#define NV_XP_LANE_ERROR_STATUS_SYNC_HDR_ORDER_ERR_NOT_ACTIVE   0x00000000 /* R-I-V */
#define NV_XP_LANE_ERROR_STATUS_SYNC_HDR_ORDER_ERR_ACTIVE       0x00000001 /* R---V */
#define NV_XP_LANE_ERROR_STATUS_OS_DATA_SEQ_ERR                        2:2 /* RWIVF */
#define NV_XP_LANE_ERROR_STATUS_OS_DATA_SEQ_ERR_NOT_ACTIVE      0x00000000 /* R-I-V */
#define NV_XP_LANE_ERROR_STATUS_OS_DATA_SEQ_ERR_ACTIVE          0x00000001 /* R---V */
#define NV_XP_LANE_ERROR_STATUS_TSX_DATA_SEQ_ERR                       3:3 /* RWIVF */
#define NV_XP_LANE_ERROR_STATUS_TSX_DATA_SEQ_ERR_NOT_ACTIVE     0x00000000 /* R-I-V */
#define NV_XP_LANE_ERROR_STATUS_TSX_DATA_SEQ_ERR_ACTIVE         0x00000001 /* R---V */
#define NV_XP_LANE_ERROR_STATUS_SKPOS_LFSR_ERR                         4:4 /* RWIVF */
#define NV_XP_LANE_ERROR_STATUS_SKPOS_LFSR_ERR_NOT_ACTIVE       0x00000000 /* R-I-V */
#define NV_XP_LANE_ERROR_STATUS_SKPOS_LFSR_ERR_ACTIVE           0x00000001 /* R---V */
#define NV_XP_LANE_ERROR_STATUS_RX_CLK_FIFO_OVERFLOW                   5:5 /* RWIVF */
#define NV_XP_LANE_ERROR_STATUS_RX_CLK_FIFO_OVERFLOW_NOT_ACTIVE 0x00000000 /* R-I-V */
#define NV_XP_LANE_ERROR_STATUS_RX_CLK_FIFO_OVERFLOW_ACTIVE     0x00000001 /* R---V */
#define NV_XP_LANE_ERROR_STATUS_ELASTIC_FIFO_OVERFLOW                 6:6 /* RWIVF */
#define NV_XP_LANE_ERROR_STATUS_ELASTIC_FIFO_OVERFLOW_NOT_ACTIVE 0x00000000 /* R-I-V */
#define NV_XP_LANE_ERROR_STATUS_ELASTIC_FIFO_OVERFLOW_ACTIVE     0x00000001 /* R---V */
#define NV_XP_LANE_ERROR_STATUS_RCVD_LINK_NUM_ERR                      7:7 /* RWIVF */
#define NV_XP_LANE_ERROR_STATUS_RCVD_LINK_NUM_ERR_NOT_ACTIVE    0x00000000 /* R-I-V */
#define NV_XP_LANE_ERROR_STATUS_RCVD_LINK_NUM_ERR_ACTIVE        0x00000000 /* R---V */
#define NV_XP_LANE_ERROR_STATUS_RCVD_LANE_NUM_ERR                      8:8 /* RWIVF */
#define NV_XP_LANE_ERROR_STATUS_RCVD_LANE_NUM_ERR_NOT_ACTIVE    0x00000000 /* R-I-V */
#define NV_XP_LANE_ERROR_STATUS_RCVD_LANE_NUM_ERR_ACTIVE        0x00000000 /* R---V */
#define NV_XP_LANE_ERROR_STATUS_SKP_RCV_SYM_ERR                        9:9 /* RWIVF */
#define NV_XP_LANE_ERROR_STATUS_SKP_RCV_SYM_ERR_NOT_ACTIVE      0x00000000 /* R-I-V */
#define NV_XP_LANE_ERROR_STATUS_SKP_RCV_SYM_ERR_ACTIVE          0x00000001 /* R---V */
#define NV_XP_LANE_ERROR_STATUS_SKPOS_PARITY_ERR                     10:10 /* RWIVF */
#define NV_XP_LANE_ERROR_STATUS_SKPOS_PARITY_ERR_NOT_ACTIVE     0x00000000 /* R-I-V */
#define NV_XP_LANE_ERROR_STATUS_SKPOS_PARITY_ERR_ACTIVE         0x00000001 /* R---V */
#define NV_XP_LANE_ERROR_STATUS_CTLSKPOS_LOCAL_DATA_PARITY_ERR                            11:11 /* RWIVF */
#define NV_XP_LANE_ERROR_STATUS_CTLSKPOS_LOCAL_DATA_PARITY_ERR_NOT_ACTIVE            0x00000000 /* R-I-V */
#define NV_XP_LANE_ERROR_STATUS_CTLSKPOS_LOCAL_DATA_PARITY_ERR_ACTIVE                0x00000001 /* R---V */
#define NV_XP_LANE_ERROR_STATUS_CTLSKPOS_FIRST_RETIMER_DATA_PARITY_ERR                    12:12 /* RWIVF */
#define NV_XP_LANE_ERROR_STATUS_CTLSKPOS_FIRST_RETIMER_DATA_PARITY_ERR_NOT_ACTIVE    0x00000000 /* R-I-V */
#define NV_XP_LANE_ERROR_STATUS_CTLSKPOS_FIRST_RETIMER_DATA_PARITY_ERR_ACTIVE        0x00000001 /* R---V */
#define NV_XP_LANE_ERROR_STATUS_CTLSKPOS_SECOND_RETIMER_DATA_PARITY_ERR                   13:13 /* RWIVF */
#define NV_XP_LANE_ERROR_STATUS_CTLSKPOS_SECOND_RETIMER_DATA_PARITY_ERR_NOT_ACTIVE   0x00000000 /* R-I-V */
#define NV_XP_LANE_ERROR_STATUS_CTLSKPOS_SECOND_RETIMER_DATA_PARITY_ERR_ACTIVE       0x00000001 /* R---V */
#define NV_XP_LANE_ERROR_STATUS_CTLSKPOS_MARGIN_CRC_ERR                                   14:14 /* RWIVF */
#define NV_XP_LANE_ERROR_STATUS_CTLSKPOS_MARGIN_CRC_ERR_NOT_ACTIVE                   0x00000000 /* R-I-V */
#define NV_XP_LANE_ERROR_STATUS_CTLSKPOS_MARGIN_CRC_ERR_ACTIVE                       0x00000001 /* R---V */
#define NV_XP_LANE_ERROR_STATUS_CTLSKPOS_MARGIN_PARITY_ERR                                15:15 /* RWIVF */
#define NV_XP_LANE_ERROR_STATUS_CTLSKPOS_MARGIN_PARITY_ERR_NOT_ACTIVE                0x00000000 /* R-I-V */
#define NV_XP_LANE_ERROR_STATUS_CTLSKPOS_MARGIN_PARITY_ERR_ACTIVE                    0x00000001 /* R---V */

#define NV_XP_LANE_ERRORS_COUNT_0                                   0x0008D40C /* R--4R */
#define NV_XP_LANE_ERRORS_COUNT_0_LANE_0_VALUE                             7:0 /* R-IVF */
#define NV_XP_LANE_ERRORS_COUNT_0_LANE_0_VALUE_INIT                 0x00000000 /* R-I-V */
#define NV_XP_LANE_ERRORS_COUNT_0_LANE_1_VALUE                            15:8 /* R-IVF */
#define NV_XP_LANE_ERRORS_COUNT_0_LANE_1_VALUE_INIT                 0x00000000 /* R-I-V */
#define NV_XP_LANE_ERRORS_COUNT_0_LANE_2_VALUE                           23:16 /* R-IVF */
#define NV_XP_LANE_ERRORS_COUNT_0_LANE_2_VALUE_INIT                 0x00000000 /* R-I-V */
#define NV_XP_LANE_ERRORS_COUNT_0_LANE_3_VALUE                           31:24 /* R-IVF */
#define NV_XP_LANE_ERRORS_COUNT_0_LANE_3_VALUE_INIT                 0x00000000 /* R-I-V */

#define NV_XP_LANE_ERRORS_COUNT_1                                   0x0008D410 /* R--4R */
#define NV_XP_LANE_ERRORS_COUNT_1_LANE_4_VALUE                             7:0 /* R-IVF */
#define NV_XP_LANE_ERRORS_COUNT_1_LANE_4_VALUE_INIT                 0x00000000 /* R-I-V */
#define NV_XP_LANE_ERRORS_COUNT_1_LANE_5_VALUE                            15:8 /* R-IVF */
#define NV_XP_LANE_ERRORS_COUNT_1_LANE_5_VALUE_INIT                 0x00000000 /* R-I-V */
#define NV_XP_LANE_ERRORS_COUNT_1_LANE_6_VALUE                           23:16 /* R-IVF */
#define NV_XP_LANE_ERRORS_COUNT_1_LANE_6_VALUE_INIT                 0x00000000 /* R-I-V */
#define NV_XP_LANE_ERRORS_COUNT_1_LANE_7_VALUE                           31:24 /* R-IVF */
#define NV_XP_LANE_ERRORS_COUNT_1_LANE_7_VALUE_INIT                 0x00000000 /* R-I-V */

#define NV_XP_LANE_ERRORS_COUNT_2                                   0x0008D414 /* R--4R */
#define NV_XP_LANE_ERRORS_COUNT_2_LANE_8_VALUE                             7:0 /* R-IVF */
#define NV_XP_LANE_ERRORS_COUNT_2_LANE_8_VALUE_INIT                 0x00000000 /* R-I-V */
#define NV_XP_LANE_ERRORS_COUNT_2_LANE_9_VALUE                            15:8 /* R-IVF */
#define NV_XP_LANE_ERRORS_COUNT_2_LANE_9_VALUE_INIT                 0x00000000 /* R-I-V */
#define NV_XP_LANE_ERRORS_COUNT_2_LANE_10_VALUE                          23:16 /* R-IVF */
#define NV_XP_LANE_ERRORS_COUNT_2_LANE_10_VALUE_INIT                0x00000000 /* R-I-V */
#define NV_XP_LANE_ERRORS_COUNT_2_LANE_11_VALUE                          31:24 /* R-IVF */
#define NV_XP_LANE_ERRORS_COUNT_2_LANE_11_VALUE_INIT                0x00000000 /* R-I-V */

#define NV_XP_LANE_ERRORS_COUNT_3                                   0x0008D418 /* R--4R */
#define NV_XP_LANE_ERRORS_COUNT_3_LANE_12_VALUE                            7:0 /* R-IVF */
#define NV_XP_LANE_ERRORS_COUNT_3_LANE_12_VALUE_INIT                0x00000000 /* R-I-V */
#define NV_XP_LANE_ERRORS_COUNT_3_LANE_13_VALUE                           15:8 /* R-IVF */
#define NV_XP_LANE_ERRORS_COUNT_3_LANE_13_VALUE_INIT                0x00000000 /* R-I-V */
#define NV_XP_LANE_ERRORS_COUNT_3_LANE_14_VALUE                          23:16 /* R-IVF */
#define NV_XP_LANE_ERRORS_COUNT_3_LANE_14_VALUE_INIT                0x00000000 /* R-I-V */
#define NV_XP_LANE_ERRORS_COUNT_3_LANE_15_VALUE                          31:24 /* R-IVF */
#define NV_XP_LANE_ERRORS_COUNT_3_LANE_15_VALUE_INIT                0x00000000 /* R-I-V */

#define NV_XP_L1_1_ENTRY_COUNT(i)                           (0x0008D910+(i)*4) /* R--4A */
#define NV_XP_L1_1_ENTRY_COUNT__SIZE_1                                       1 /*       */
#define NV_XP_L1_1_ENTRY_COUNT_VALUE                                      31:0 /* R-IVF */
#define NV_XP_L1_1_ENTRY_COUNT_VALUE_INIT                           0x00000000 /* R-I-V */

#define NV_XP_L1_2_ENTRY_COUNT(i)                           (0x0008D950+(i)*4) /* R--4A */
#define NV_XP_L1_2_ENTRY_COUNT__SIZE_1                                       1 /*       */
#define NV_XP_L1_2_ENTRY_COUNT_VALUE                                      31:0 /* R-IVF */
#define NV_XP_L1_2_ENTRY_COUNT_VALUE_INIT                           0x00000000 /* R-I-V */

#define NV_XP_L1_2_ABORT_COUNT(i)                           (0x0008D990+(i)*4) /* R--4A */
#define NV_XP_L1_2_ABORT_COUNT__SIZE_1                                       1 /*       */
#define NV_XP_L1_2_ABORT_COUNT_VALUE                                      31:0 /* R-IVF */
#define NV_XP_L1_2_ABORT_COUNT_VALUE_INIT                           0x00000000 /* R-I-V */

#define NV_XP_L1_SUBSTATE_TO_DEEP_L1_TIMEOUT_COUNT(i)       (0x0008D9D0+(i)*4) /* R--4A */
#define NV_XP_L1_SUBSTATE_TO_DEEP_L1_TIMEOUT_COUNT__SIZE_1                   1 /*       */
#define NV_XP_L1_SUBSTATE_TO_DEEP_L1_TIMEOUT_COUNT_VALUE                  31:0 /* R-IVF */
#define NV_XP_L1_SUBSTATE_TO_DEEP_L1_TIMEOUT_COUNT_VALUE_INIT       0x00000000 /* R-I-V */

#define NV_XP_L1_SHORT_DURATION_COUNT(i)                    (0x0008E0C4+(i)*4) /* R--4A */
#define NV_XP_L1_SHORT_DURATION_COUNT__SIZE_1                                1 /*       */
#define NV_XP_L1_SHORT_DURATION_COUNT_VALUE                               31:0 /* R-IVF */
#define NV_XP_L1_SHORT_DURATION_COUNT_VALUE_INIT                    0x00000000 /* R-I-V */

#define NV_XP_RECEIVER_ERRORS_COUNT(i)                      (0x0008D440+(i)*4) /* R--4A */
#define NV_XP_RECEIVER_ERRORS_COUNT__SIZE_1                                  1 /*       */
#define NV_XP_RECEIVER_ERRORS_COUNT_VALUE                                 15:0 /* R-IVF */
#define NV_XP_RECEIVER_ERRORS_COUNT_VALUE_INIT                      0x00000000 /* R-I-V */

#define NV_XP_REPLAY_ROLLOVER_COUNT(i)                      (0x0008D5C0+(i)*4) /* R--4A */
#define NV_XP_REPLAY_ROLLOVER_COUNT__SIZE_1                                  1 /*       */
#define NV_XP_REPLAY_ROLLOVER_COUNT_VALUE                                 15:0 /* R-IVF */
#define NV_XP_REPLAY_ROLLOVER_COUNT_VALUE_INIT                      0x00000000 /* R-I-V */

#define NV_XP_BAD_DLLP_COUNT(i)                            (0x0008D6C0+(i)*4) /* R--4A */
#define NV_XP_BAD_DLLP_COUNT__SIZE_1                                        1 /*       */
#define NV_XP_BAD_DLLP_COUNT_VALUE                                       15:0 /* R-IVF */
#define NV_XP_BAD_DLLP_COUNT_VALUE_INIT                            0x00000000 /* R-I-V */

#define NV_XP_BAD_TLP_COUNT(i)                             (0x0008D700+(i)*4) /* R--4A */
#define NV_XP_BAD_TLP_COUNT__SIZE_1                                         1 /*       */
#define NV_XP_BAD_TLP_COUNT_VALUE                                        15:0 /* R-IVF */
#define NV_XP_BAD_TLP_COUNT_VALUE_INIT                             0x00000000 /* R-I-V */

#define NV_XP__8B10B_ERRORS_COUNT                                  0x0008D404 /* R--4R */
#define NV_XP__8B10B_ERRORS_COUNT_VALUE                                  15:0 /* R-IVF */
#define NV_XP__8B10B_ERRORS_COUNT_VALUE_INIT                       0x00000000 /* R-I-V */

#define NV_XP_SYNC_HEADER_ERRORS_COUNT                              0x0008D408 /* R--4R */
#define NV_XP_SYNC_HEADER_ERRORS_COUNT_VALUE                              15:0 /* R-IVF */
#define NV_XP_SYNC_HEADER_ERRORS_COUNT_VALUE_INIT                   0x00000000 /* R-I-V */

#define NV_XP_LCRC_ERRORS_COUNT(i)                          (0x0008D480+(i)*4) /* R--4A */
#define NV_XP_LCRC_ERRORS_COUNT__SIZE_1                                      1 /*       */
#define NV_XP_LCRC_ERRORS_COUNT_VALUE                                     15:0 /* R-IVF */
#define NV_XP_LCRC_ERRORS_COUNT_VALUE_INIT                          0x00000000 /* R-I-V */

#define NV_XP_FAILED_L0S_EXITS_COUNT(i)                     (0x0008D4C0+(i)*4) /* R--4A */
#define NV_XP_FAILED_L0S_EXITS_COUNT__SIZE_1                                 1 /*       */
#define NV_XP_FAILED_L0S_EXITS_COUNT_VALUE                                15:0 /* R-IVF */
#define NV_XP_FAILED_L0S_EXITS_COUNT_VALUE_INIT                     0x00000000 /* R-I-V */

#define NV_XP_NAKS_SENT_COUNT(i)                            (0x0008D500+(i)*4) /* R--4A */
#define NV_XP_NAKS_SENT_COUNT__SIZE_1                                        1 /*       */
#define NV_XP_NAKS_SENT_COUNT_VALUE                                       15:0 /* R-IVF */
#define NV_XP_NAKS_SENT_COUNT_VALUE_INIT                            0x00000000 /* R-I-V */

#define NV_XP_NAKS_RCVD_COUNT(i)                            (0x0008D540+(i)*4) /* R--4A */
#define NV_XP_NAKS_RCVD_COUNT__SIZE_1                                        1 /*       */
#define NV_XP_NAKS_RCVD_COUNT_VALUE                                       15:0 /* R-IVF */
#define NV_XP_NAKS_RCVD_COUNT_VALUE_INIT                            0x00000000 /* R-I-V */
#define NV_XP_NAKS_RCVD_COUNT_ILLOGICAL_VALUE                            23:16 /* R-IVF */
#define NV_XP_NAKS_RCVD_COUNT_ILLOGICAL_VALUE_INIT                  0x00000000 /* R-I-V */

#define NV_XP_L1_TO_RECOVERY_COUNT(i)                       (0x0008D600+(i)*4) /* R--4A */
#define NV_XP_L1_TO_RECOVERY_COUNT__SIZE_1                                   1 /*       */
#define NV_XP_L1_TO_RECOVERY_COUNT_VALUE                                  31:0 /* R-IVF */
#define NV_XP_L1_TO_RECOVERY_COUNT_VALUE_INIT                       0x00000000 /* R-I-V */

#define NV_XP_L0_TO_RECOVERY_COUNT(i)                       (0x0008D640+(i)*4) /* R--4A */
#define NV_XP_L0_TO_RECOVERY_COUNT__SIZE_1                                   1 /*       */
#define NV_XP_L0_TO_RECOVERY_COUNT_VALUE                                  31:0 /* R-IVF */
#define NV_XP_L0_TO_RECOVERY_COUNT_VALUE_INIT                       0x00000000 /* R-I-V */

#define NV_XP_RECOVERY_COUNT(i)                             (0x0008D680+(i)*4) /* R--4A */
#define NV_XP_RECOVERY_COUNT__SIZE_1                                         1 /*       */
#define NV_XP_RECOVERY_COUNT_VALUE                                        31:0 /* R-IVF */
#define NV_XP_RECOVERY_COUNT_VALUE_INIT                             0x00000000 /* R-I-V */

#define NV_XP_CHIPSET_XMIT_L0S_ENTRY_COUNT(i)               (0x0008D740+(i)*4) /* R--4A */
#define NV_XP_CHIPSET_XMIT_L0S_ENTRY_COUNT__SIZE_1                           1 /*       */
#define NV_XP_CHIPSET_XMIT_L0S_ENTRY_COUNT_VALUE                          31:0 /* R-IVF */
#define NV_XP_CHIPSET_XMIT_L0S_ENTRY_COUNT_VALUE_INIT               0x00000000 /* R-I-V */

#define NV_XP_GPU_XMIT_L0S_ENTRY_COUNT(i)                   (0x0008D780+(i)*4) /* R--4A */
#define NV_XP_GPU_XMIT_L0S_ENTRY_COUNT__SIZE_1                               1 /*       */
#define NV_XP_GPU_XMIT_L0S_ENTRY_COUNT_VALUE                              31:0 /* R-IVF */
#define NV_XP_GPU_XMIT_L0S_ENTRY_COUNT_VALUE_INIT                   0x00000000 /* R-I-V */

#define NV_XP_L1_ENTRY_COUNT(i)                             (0x0008D7C0+(i)*4) /* R--4A */
#define NV_XP_L1_ENTRY_COUNT__SIZE_1                                         1 /*       */
#define NV_XP_L1_ENTRY_COUNT_VALUE                                        31:0 /* R-IVF */
#define NV_XP_L1_ENTRY_COUNT_VALUE_INIT                             0x00000000 /* R-I-V */

#define NV_XP_L1P_ENTRY_COUNT(i)                            (0x0008D800+(i)*4) /* R--4A */
#define NV_XP_L1P_ENTRY_COUNT__SIZE_1                                        1 /*       */
#define NV_XP_L1P_ENTRY_COUNT_VALUE                                       31:0 /* R-IVF */
#define NV_XP_L1P_ENTRY_COUNT_VALUE_INIT                            0x00000000 /* R-I-V */

#define NV_XP_DEEP_L1_ENTRY_COUNT(i)                        (0x0008D840+(i)*4) /* R--4A */
#define NV_XP_DEEP_L1_ENTRY_COUNT__SIZE_1                                    1 /*       */
#define NV_XP_DEEP_L1_ENTRY_COUNT_VALUE                                   31:0 /* R-IVF */
#define NV_XP_DEEP_L1_ENTRY_COUNT_VALUE_INIT                        0x00000000 /* R-I-V */

#define NV_XP_ASLM_COUNT(i)                                 (0x0008D880+(i)*4) /* R--4A */
#define NV_XP_ASLM_COUNT__SIZE_1                                             1 /*       */
#define NV_XP_ASLM_COUNT_VALUE                                            15:0 /* R-IVF */
#define NV_XP_ASLM_COUNT_VALUE_INIT                                 0x00000000 /* R-I-V */

#define NV_XP_ERROR_COUNTER_RESET                                        0x0008D900 /* RWI4R */
#define NV_XP_ERROR_COUNTER_RESET_8B10B_ERRORS_COUNT                            0:0 /* RWIVF */
#define NV_XP_ERROR_COUNTER_RESET_8B10B_ERRORS_COUNT_DONE                0x00000000 /* RWI-V */
#define NV_XP_ERROR_COUNTER_RESET_8B10B_ERRORS_COUNT_PENDING             0x00000001 /* -W--T */
#define NV_XP_ERROR_COUNTER_RESET_SYNC_HEADER_ERRORS_COUNT                       1:1 /* RWIVF */
#define NV_XP_ERROR_COUNTER_RESET_SYNC_HEADER_ERRORS_COUNT_DONE           0x00000000 /* RWI-V */
#define NV_XP_ERROR_COUNTER_RESET_SYNC_HEADER_ERRORS_COUNT_PENDING        0x00000001 /* -W--T */
#define NV_XP_ERROR_COUNTER_RESET_LANE_ERRORS_COUNT                              2:2 /* RWIVF */
#define NV_XP_ERROR_COUNTER_RESET_LANE_ERRORS_COUNT_DONE                  0x00000000 /* RWI-V */
#define NV_XP_ERROR_COUNTER_RESET_LANE_ERRORS_COUNT_PENDING               0x00000001 /* -W--T */
#define NV_XP_ERROR_COUNTER_RESET_RECEIVER_ERRORS_COUNT                          3:3 /* RWIVF */
#define NV_XP_ERROR_COUNTER_RESET_RECEIVER_ERRORS_COUNT_DONE              0x00000000 /* RWI-V */
#define NV_XP_ERROR_COUNTER_RESET_RECEIVER_ERRORS_COUNT_PENDING           0x00000001 /* -W--T */
#define NV_XP_ERROR_COUNTER_RESET_LCRC_ERRORS_COUNT                              4:4 /* RWIVF */
#define NV_XP_ERROR_COUNTER_RESET_LCRC_ERRORS_COUNT_DONE                  0x00000000 /* RWI-V */
#define NV_XP_ERROR_COUNTER_RESET_LCRC_ERRORS_COUNT_PENDING               0x00000001 /* -W--T */
#define NV_XP_ERROR_COUNTER_RESET_FAILED_L0S_EXITS_COUNT                         5:5 /* RWIVF */
#define NV_XP_ERROR_COUNTER_RESET_FAILED_L0S_EXITS_COUNT_DONE             0x00000000 /* RWI-V */
#define NV_XP_ERROR_COUNTER_RESET_FAILED_L0S_EXITS_COUNT_PENDING          0x00000001 /* -W--T */
#define NV_XP_ERROR_COUNTER_RESET_NAKS_SENT_COUNT                                6:6 /* RWIVF */
#define NV_XP_ERROR_COUNTER_RESET_NAKS_SENT_COUNT_DONE                    0x00000000 /* RWI-V */
#define NV_XP_ERROR_COUNTER_RESET_NAKS_SENT_COUNT_PENDING                 0x00000001 /* -W--T */
#define NV_XP_ERROR_COUNTER_RESET_NAKS_RCVD_COUNT                                7:7 /* RWIVF */
#define NV_XP_ERROR_COUNTER_RESET_NAKS_RCVD_COUNT_DONE                    0x00000000 /* RWI-V */
#define NV_XP_ERROR_COUNTER_RESET_NAKS_RCVD_COUNT_PENDING                 0x00000001 /* -W--T */
#define NV_XP_ERROR_COUNTER_RESET_REPLAY_COUNT                                   8:8 /* RWIVF */
#define NV_XP_ERROR_COUNTER_RESET_REPLAY_COUNT_DONE                       0x00000000 /* RWI-V */
#define NV_XP_ERROR_COUNTER_RESET_REPLAY_COUNT_PENDING                    0x00000001 /* -W--T */
#define NV_XP_ERROR_COUNTER_RESET_REPLAY_ROLLOVER_COUNT                          9:9 /* RWIVF */
#define NV_XP_ERROR_COUNTER_RESET_REPLAY_ROLLOVER_COUNT_DONE              0x00000000 /* RWI-V */
#define NV_XP_ERROR_COUNTER_RESET_REPLAY_ROLLOVER_COUNT_PENDING           0x00000001 /* -W--T */
#define NV_XP_ERROR_COUNTER_RESET_L1_TO_RECOVERY_COUNT                         10:10 /* RWIVF */
#define NV_XP_ERROR_COUNTER_RESET_L1_TO_RECOVERY_COUNT_DONE               0x00000000 /* RWI-V */
#define NV_XP_ERROR_COUNTER_RESET_L1_TO_RECOVERY_COUNT_PENDING            0x00000001 /* -W--T */
#define NV_XP_ERROR_COUNTER_RESET_L0_TO_RECOVERY_COUNT                         11:11 /* RWIVF */
#define NV_XP_ERROR_COUNTER_RESET_L0_TO_RECOVERY_COUNT_DONE               0x00000000 /* RWI-V */
#define NV_XP_ERROR_COUNTER_RESET_L0_TO_RECOVERY_COUNT_PENDING            0x00000001 /* -W--T */
#define NV_XP_ERROR_COUNTER_RESET_RECOVERY_COUNT                               12:12 /* RWIVF */
#define NV_XP_ERROR_COUNTER_RESET_RECOVERY_COUNT_DONE                     0x00000000 /* RWI-V */
#define NV_XP_ERROR_COUNTER_RESET_RECOVERY_COUNT_PENDING                  0x00000001 /* -W--T */
#define NV_XP_ERROR_COUNTER_RESET_BAD_DLLP_COUNT                               13:13 /* RWIVF */
#define NV_XP_ERROR_COUNTER_RESET_BAD_DLLP_COUNT_DONE                     0x00000000 /* RWI-V */
#define NV_XP_ERROR_COUNTER_RESET_BAD_DLLP_COUNT_PENDING                  0x00000001 /* -W--T */
#define NV_XP_ERROR_COUNTER_RESET_BAD_TLP_COUNT                                14:14 /* RWIVF */
#define NV_XP_ERROR_COUNTER_RESET_BAD_TLP_COUNT_DONE                      0x00000000 /* RWI-V */
#define NV_XP_ERROR_COUNTER_RESET_BAD_TLP_COUNT_PENDING                   0x00000001 /* -W--T */
#define NV_XP_ERROR_COUNTER_RESET_CHIPSET_XMIT_L0S_ENTRY_COUNT                 15:15 /* RWIVF */
#define NV_XP_ERROR_COUNTER_RESET_CHIPSET_XMIT_L0S_ENTRY_COUNT_DONE       0x00000000 /* RWI-V */
#define NV_XP_ERROR_COUNTER_RESET_CHIPSET_XMIT_L0S_ENTRY_COUNT_PENDING    0x00000001 /* -W--T */
#define NV_XP_ERROR_COUNTER_RESET_GPU_XMIT_L0S_ENTRY_COUNT                     16:16 /* RWIVF */
#define NV_XP_ERROR_COUNTER_RESET_GPU_XMIT_L0S_ENTRY_COUNT_DONE           0x00000000 /* RWI-V */
#define NV_XP_ERROR_COUNTER_RESET_GPU_XMIT_L0S_ENTRY_COUNT_PENDING        0x00000001 /* -W--T */
#define NV_XP_ERROR_COUNTER_RESET_L1_ENTRY_COUNT                               17:17 /* RWIVF */
#define NV_XP_ERROR_COUNTER_RESET_L1_ENTRY_COUNT_DONE                     0x00000000 /* RWI-V */
#define NV_XP_ERROR_COUNTER_RESET_L1_ENTRY_COUNT_PENDING                  0x00000001 /* -W--T */
#define NV_XP_ERROR_COUNTER_RESET_L1P_ENTRY_COUNT                              18:18 /* RWIVF */
#define NV_XP_ERROR_COUNTER_RESET_L1P_ENTRY_COUNT_DONE                    0x00000000 /* RWI-V */
#define NV_XP_ERROR_COUNTER_RESET_L1P_ENTRY_COUNT_PENDING                 0x00000001 /* -W--T */
#define NV_XP_ERROR_COUNTER_RESET_DEEP_L1_ENTRY_COUNT                          19:19 /* RWIVF */
#define NV_XP_ERROR_COUNTER_RESET_DEEP_L1_ENTRY_COUNT_DONE                0x00000000 /* RWI-V */
#define NV_XP_ERROR_COUNTER_RESET_DEEP_L1_ENTRY_COUNT_PENDING             0x00000001 /* -W--T */
#define NV_XP_ERROR_COUNTER_RESET_ASLM_COUNT                                   20:20 /* RWIVF */
#define NV_XP_ERROR_COUNTER_RESET_ASLM_COUNT_DONE                         0x00000000 /* RWI-V */
#define NV_XP_ERROR_COUNTER_RESET_ASLM_COUNT_PENDING                      0x00000001 /* -W--T */
#define NV_XP_ERROR_COUNTER_RESET_SKPOS_ERRORS_COUNT                           21:21 /* RWIVF */
#define NV_XP_ERROR_COUNTER_RESET_SKPOS_ERRORS_COUNT_DONE                 0x00000000 /* RWI-V */
#define NV_XP_ERROR_COUNTER_RESET_SKPOS_ERRORS_COUNT_PENDING              0x00000001 /* -W--T */
#define NV_XP_ERROR_COUNTER_RESET_L1_1_ENTRY_COUNT                             22:22 /* RWIVF */
#define NV_XP_ERROR_COUNTER_RESET_L1_1_ENTRY_COUNT_DONE                   0x00000000 /* RWI-V */
#define NV_XP_ERROR_COUNTER_RESET_L1_1_ENTRY_COUNT_PENDING                0x00000001 /* -W--T */
#define NV_XP_ERROR_COUNTER_RESET_L1_2_ENTRY_COUNT                             23:23 /* RWIVF */
#define NV_XP_ERROR_COUNTER_RESET_L1_2_ENTRY_COUNT_DONE                   0x00000000 /* RWI-V */
#define NV_XP_ERROR_COUNTER_RESET_L1_2_ENTRY_COUNT_PENDING                0x00000001 /* -W--T */
#define NV_XP_ERROR_COUNTER_RESET_L1_2_ABORT_COUNT                             24:24 /* RWIVF */
#define NV_XP_ERROR_COUNTER_RESET_L1_2_ABORT_COUNT_DONE                   0x00000000 /* RWI-V */
#define NV_XP_ERROR_COUNTER_RESET_L1_2_ABORT_COUNT_PENDING                0x00000001 /* -W--T */
#define NV_XP_ERROR_COUNTER_RESET_L1_SUBSTATE_TO_DEEP_L1_TIMEOUT_COUNT                         25:25 /* RWIVF */
#define NV_XP_ERROR_COUNTER_RESET_L1_SUBSTATE_TO_DEEP_L1_TIMEOUT_COUNT_DONE               0x00000000 /* RWI-V */
#define NV_XP_ERROR_COUNTER_RESET_L1_SUBSTATE_TO_DEEP_L1_TIMEOUT_COUNT_PENDING            0x00000001 /* -W--T */
#define NV_XP_ERROR_COUNTER_RESET_L1_SHORT_DURATION_COUNT                      26:26 /* RWIVF */
#define NV_XP_ERROR_COUNTER_RESET_L1_SHORT_DURATION_COUNT_DONE            0x00000000 /* RWI-V */
#define NV_XP_ERROR_COUNTER_RESET_L1_SHORT_DURATION_COUNT_PENDING         0x00000001 /* -W--T */

#define NV_XP_PRI_XP3G_CG                          0x0008E000 /* RWI4R */
#define NV_XP_PRI_XP3G_CG_IDLE_CG_DLY_CNT                 5:0 /* RWIVF */
#define NV_XP_PRI_XP3G_CG_IDLE_CG_DLY_CNT_HWINIT   0x00000000 /* RWI-V */
#define NV_XP_PRI_XP3G_CG_IDLE_CG_DLY_CNT__PROD    0x0000000B /* RW--V */
#define NV_XP_PRI_XP3G_CG_IDLE_CG_EN                      6:6 /* RWIVF */
#define NV_XP_PRI_XP3G_CG_IDLE_CG_EN_ENABLED       0x00000001 /* RW--V */
#define NV_XP_PRI_XP3G_CG_IDLE_CG_EN_DISABLED      0x00000000 /* RWI-V */
#define NV_XP_PRI_XP3G_CG_IDLE_CG_EN__PROD         0x00000001 /* RW--V */
#define NV_XP_PRI_XP3G_CG_STATE_CG_EN                     7:7 /*       */
#define NV_XP_PRI_XP3G_CG_STATE_CG_EN_ENABLED      0x00000001 /*       */
#define NV_XP_PRI_XP3G_CG_STATE_CG_EN_DISABLED     0x00000000 /*       */
#define NV_XP_PRI_XP3G_CG_STATE_CG_EN__PROD        0x00000000 /*       */
#define NV_XP_PRI_XP3G_CG_STALL_CG_DLY_CNT               13:8 /*       */
#define NV_XP_PRI_XP3G_CG_STALL_CG_DLY_CNT_HWINIT  0x00000000 /*       */
#define NV_XP_PRI_XP3G_CG_STALL_CG_DLY_CNT__PROD   0x00000000 /*       */
#define NV_XP_PRI_XP3G_CG_STALL_CG_EN                   14:14 /* RWIVF */
#define NV_XP_PRI_XP3G_CG_STALL_CG_EN_ENABLED      0x00000001 /* RW--V */
#define NV_XP_PRI_XP3G_CG_STALL_CG_EN_DISABLED     0x00000000 /* RWI-V */
#define NV_XP_PRI_XP3G_CG_STALL_CG_EN__PROD        0x00000000 /* RW--V */
#define NV_XP_PRI_XP3G_CG_QUIESCENT_CG_EN               15:15 /*       */
#define NV_XP_PRI_XP3G_CG_QUIESCENT_CG_EN_ENABLED  0x00000001 /*       */
#define NV_XP_PRI_XP3G_CG_QUIESCENT_CG_EN_DISABLED 0x00000000 /*       */
#define NV_XP_PRI_XP3G_CG_QUIESCENT_CG_EN__PROD    0x00000000 /*       */
#define NV_XP_PRI_XP3G_CG_WAKEUP_DLY_CNT                19:16 /* RWIVF */
#define NV_XP_PRI_XP3G_CG_WAKEUP_DLY_CNT_HWINIT    0x00000000 /* RWI-V */
#define NV_XP_PRI_XP3G_CG_WAKEUP_DLY_CNT__PROD     0x00000000 /* RW--V */
#define NV_XP_PRI_XP3G_CG_THROT_CLK_CNT                 23:20 /*       */
#define NV_XP_PRI_XP3G_CG_THROT_CLK_CNT_FULLSPEED  0x0000000f /*       */
#define NV_XP_PRI_XP3G_CG_THROT_CLK_CNT__PROD      0x00000000 /*       */
#define NV_XP_PRI_XP3G_CG_DI_DT_SKEW_VAL                27:24 /*       */
#define NV_XP_PRI_XP3G_CG_DI_DT_SKEW_VAL_HWINIT    0x00000000 /*       */
#define NV_XP_PRI_XP3G_CG_DI_DT_SKEW_VAL__PROD     0x00000000 /*       */
#define NV_XP_PRI_XP3G_CG_THROT_CLK_EN                  28:28 /*       */
#define NV_XP_PRI_XP3G_CG_THROT_CLK_EN_ENABLED     0x00000001 /*       */
#define NV_XP_PRI_XP3G_CG_THROT_CLK_EN_DISABLED    0x00000000 /*       */
#define NV_XP_PRI_XP3G_CG_THROT_CLK_EN__PROD       0x00000000 /*       */
#define NV_XP_PRI_XP3G_CG_THROT_CLK_SW_OVER             29:29 /*       */
#define NV_XP_PRI_XP3G_CG_THROT_CLK_SW_OVER_EN     0x00000001 /*       */
#define NV_XP_PRI_XP3G_CG_THROT_CLK_SW_OVER_DIS    0x00000000 /*       */
#define NV_XP_PRI_XP3G_CG_THROT_CLK_SW_OVER__PROD  0x00000000 /*       */
#define NV_XP_PRI_XP3G_CG_PAUSE_CG_EN                   30:30 /*       */
#define NV_XP_PRI_XP3G_CG_PAUSE_CG_EN_ENABLED      0x00000001 /*       */
#define NV_XP_PRI_XP3G_CG_PAUSE_CG_EN_DISABLED     0x00000000 /*       */
#define NV_XP_PRI_XP3G_CG_PAUSE_CG_EN__PROD        0x00000000 /*       */
#define NV_XP_PRI_XP3G_CG_HALT_CG_EN                    31:31 /*       */
#define NV_XP_PRI_XP3G_CG_HALT_CG_EN_ENABLED       0x00000001 /*       */
#define NV_XP_PRI_XP3G_CG_HALT_CG_EN_DISABLED      0x00000000 /*       */
#define NV_XP_PRI_XP3G_CG_HALT_CG_EN__PROD         0x00000000 /*       */

#define NV_XP_PRI_XP3G_CG1                               0x0008E004 /* RWI4R */
#define NV_XP_PRI_XP3G_CG1_MONITOR_CG_EN                      0:0 /* RWIVF */
#define NV_XP_PRI_XP3G_CG1_MONITOR_CG_EN_ENABLED       0x00000001 /* RW--V */
#define NV_XP_PRI_XP3G_CG1_MONITOR_CG_EN_DISABLED      0x00000000 /* RWI-V */
#define NV_XP_PRI_XP3G_CG1_MONITOR_CG_EN__PROD         0x00000000 /* RW--V */

#define NV_XP_REPLAY_COUNT(i)                               (0x0008D580+(i)*4) /* R--4A */
#define NV_XP_REPLAY_COUNT__SIZE_1                    1 /*       */
#define NV_XP_REPLAY_COUNT_VALUE                                          31:0 /* R-IVF */
#define NV_XP_REPLAY_COUNT_VALUE_INIT                               0x00000000 /* R-I-V */
#endif // __lr10_dev_nv_xp_h__
