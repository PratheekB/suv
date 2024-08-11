/*
 * SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "rmapi/client.h"
#include "gpu/mmu/uvm_sw.h"
#include "virtualization/hypervisor/hypervisor.h"
#include "rmapi/client.h"

void
uvmswInitSwMethodState_IMPL
(
    UvmSwObject *pUvmSw
)
{
    pUvmSw->methodA = 0;
    pUvmSw->methodB = 0;
    pUvmSw->bCancelMethodASet = NV_FALSE;
    pUvmSw->bCancelMethodBSet = NV_FALSE;
    pUvmSw->bClearMethodASet = NV_FALSE;
}

NV_STATUS
uvmswConstruct_IMPL
(
    UvmSwObject                  *pUvmSw,
    CALL_CONTEXT                 *pCallContext,
    RS_RES_ALLOC_PARAMS_INTERNAL *pParams
)
{
    NvHandle hClient = pCallContext->pClient->hClient;
    RmClient *pRmClient = dynamicCast(pCallContext->pClient, RmClient);
    RS_PRIV_LEVEL privLevel = pCallContext->secInfo.privLevel;

    if (!(rmclientIsAdmin(pRmClient, privLevel) || hypervisorCheckForObjectAccess(hClient)))
        return NV_ERR_INVALID_CLIENT;

    uvmswInitSwMethodState(pUvmSw);

    return NV_OK;
}

void
uvmswDestruct_IMPL
(
    UvmSwObject *pUvmSw
)
{
    ChannelDescendant *pChannelDescendant = staticCast(pUvmSw, ChannelDescendant);

    chandesIsolateOnDestruct(pChannelDescendant);
}
