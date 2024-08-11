/*
 * SPDX-FileCopyrightText: Copyright (c) 2003-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef __gp102_dev_fb_h__
#define __gp102_dev_fb_h__
#define NV_PFB_PRI_MMU_LOCAL_MEMORY_RANGE                           0x00100CE0 /* RW-4R */
#define NV_PFB_PRI_MMU_LOCAL_MEMORY_RANGE_LOWER_SCALE                      3:0 /* RWEVF */
#define NV_PFB_PRI_MMU_LOCAL_MEMORY_RANGE_LOWER_MAG                        9:4 /* RWEVF */
#define NV_PFB_PRI_MMU_LOCAL_MEMORY_RANGE_ECC_MODE                       30:30 /* RWEVF */
#define NV_PFB_PRI_MMU_LOCAL_MEMORY_RANGE_ECC_MODE_DISABLED         0x00000000 /* RWE-V */
#define NV_PFB_PRI_MMU_LOCAL_MEMORY_RANGE_ECC_MODE_ENABLED          0x00000001 /* RW--V */
#endif // __gp102_dev_fb_h__
