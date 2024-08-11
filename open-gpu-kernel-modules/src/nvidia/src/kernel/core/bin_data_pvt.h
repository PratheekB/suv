/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2019 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef _BINDATA_PRIVATE_H
#define _BINDATA_PRIVATE_H

#include "core/core.h"
#include "lib/zlib/inflate.h"
#include "core/bin_data.h"

/**************************************************************************************************************
*
*    File:  bin_data_private.h
*
*    Description:
*        Private data structure for binary data management API
*
**************************************************************************************************************/

//
// WARNING: This header should not be included directly (outside of bindata
// impl)
//
// TODO: Clean up the references that have snuck in and move outside of the
// public module directory
//


//
// Private data structure for binary data management
//

//
// Binary data management static information (generated by bindata.pl)
//
typedef struct
{
    NvU32           actualSize;         // size of (uncompressed) pData
    NvU32           compressedSize;     // size of (compressed) pData array
    const void *    pData;              // pointer to the raw binary (whether compressed or not) data
    NvBool          bCompressed            : 1;    // is compressed?
    NvBool          bFileOverrideSupported : 1;    // contain information for file overriding?
    NvBool          bReferenced            : 1;    // Has this data been referenced before?
} BINDATA_STORAGE_PVT, *PBINDATA_STORAGE_PVT;

//
// Binary data management runtime information
//
struct BINDATA_RUNTIME_INFO
{
    const BINDATA_STORAGE_PVT  *pBinStoragePvt;  // pointer to the static init struct
    PGZ_INFLATE_STATE           pGzState;        // used by gzip
    NvU32                       currDataPos;     // position where next chunk acquire should start at
};

//
// This knob controls whether the data will be placed into .rodata section and
// be considered constant for the lifetime of RM, or if it can be modified
// during execution. Right now, we only need to modify it on GSP to reclaim
// the memory as general purpose heap.
//
#define BINDATA_IS_MUTABLE RMCFG_FEATURE_PLATFORM_GSP
#if BINDATA_IS_MUTABLE
#define BINDATA_CONST
#else
#define BINDATA_CONST const
#endif

void bindataMarkReferenced(const BINDATA_STORAGE *pBinStorage);

#endif // _BINDATA_PRIVATE_H
