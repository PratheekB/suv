/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "deprecated/rmapi_deprecated.h"

#include "finn_rm_api.h"
#include "gpu/gpu.h"
#include "core/locks.h"

/*!
 * Some clients are still making these legacy GSS controls. We no longer support these in RM,
 * but until all the numerous tools are updated to use alternative APIs, just forward all of them
 * to GSP and let it deal with what is or isn't valid.
*/
NV_STATUS RmGssLegacyRpcCmd
(
    API_SECURITY_INFO  *pSecInfo,
    DEPRECATED_CONTEXT *pContextInternal,
    NVOS54_PARAMETERS  *pArgs
)
{
    OBJGPU    *pGpu           = NULL;
    RsClient  *pClient        = NULL;
    NV_STATUS  status         = NV_OK;
    GPU_MASK   gpuMaskRelease = 0;
    void      *pKernelParams  = NULL;

    NV_ASSERT_OR_RETURN((pArgs->cmd & RM_GSS_LEGACY_MASK),
                        NV_ERR_INVALID_STATE);

    if (((pArgs->cmd & RM_GSS_LEGACY_MASK_PRIVILEGED) == RM_GSS_LEGACY_MASK_PRIVILEGED) &&
        (pSecInfo->privLevel < RS_PRIV_LEVEL_USER_ROOT))
    {
        return NV_ERR_INSUFFICIENT_PERMISSIONS;
    }

    NV_CHECK_OK_OR_ELSE(status,
                        LEVEL_ERROR,
                        serverGetClientUnderLock(&g_resServ, pArgs->hClient, &pClient),
                        return NV_ERR_INVALID_ARGUMENT);

    NV_CHECK_OK_OR_ELSE(status,
                        LEVEL_ERROR,
                        gpuGetByHandle(pClient, pArgs->hObject, NULL, &pGpu),
                        return NV_ERR_INVALID_ARGUMENT);

    RM_API *pRmApi = GPU_GET_PHYSICAL_RMAPI(pGpu);

    if (pSecInfo->paramLocation == PARAM_LOCATION_USER)
    {
        pKernelParams = portMemAllocNonPaged(pArgs->paramsSize);
        if (pKernelParams == NULL)
        {
            status = NV_ERR_NO_MEMORY;
            goto done;
        }

        status = portMemExCopyFromUser(pArgs->params, pKernelParams, pArgs->paramsSize);
        if (status != NV_OK)
            goto done;
    }
    else
    {
        pKernelParams = (void*)pArgs->params;
    }

    status = rmGpuGroupLockAcquire(pGpu->gpuInstance,
                           GPU_LOCK_GRP_SUBDEVICE,
                           GPUS_LOCK_FLAGS_NONE,
                           RM_LOCK_MODULES_RPC,
                           &gpuMaskRelease);
    if (status != NV_OK)
        goto done;

    status = pRmApi->Control(pRmApi,
                             pArgs->hClient,
                             pArgs->hObject,
                             pArgs->cmd,
                             pKernelParams,
                             pArgs->paramsSize);

done:
    if (gpuMaskRelease != 0)
    {
        rmGpuGroupLockRelease(gpuMaskRelease, GPUS_LOCK_FLAGS_NONE);
    }

    if (pSecInfo->paramLocation == PARAM_LOCATION_USER)
    {
        if (status == NV_OK)
        {
            status = portMemExCopyToUser(pKernelParams, pArgs->params, pArgs->paramsSize);
        }
        portMemFree(pKernelParams);
    }

    return status;
}
