#define NVOC_REF_COUNT_H_PRIVATE_ACCESS_ALLOWED
#include "nvoc/runtime.h"
#include "nvoc/rtti.h"
#include "nvtypes.h"
#include "nvport/nvport.h"
#include "nvport/inline/util_valist.h"
#include "utils/nvassert.h"
#include "g_ref_count_nvoc.h"

#ifdef DEBUG
char __nvoc_class_id_uniqueness_check_0xf89281 = 1;
#endif

extern const struct NVOC_CLASS_DEF __nvoc_class_def_OBJREFCNT;

extern const struct NVOC_CLASS_DEF __nvoc_class_def_Object;

void __nvoc_init_OBJREFCNT(OBJREFCNT*);
void __nvoc_init_funcTable_OBJREFCNT(OBJREFCNT*);
NV_STATUS __nvoc_ctor_OBJREFCNT(OBJREFCNT*, Dynamic * arg_pParent, NvU32 arg_tag, RefcntStateChangeCallback * arg_pStateChangeCallback, RefcntResetCallback * arg_pResetCallback);
void __nvoc_init_dataField_OBJREFCNT(OBJREFCNT*);
void __nvoc_dtor_OBJREFCNT(OBJREFCNT*);
extern const struct NVOC_EXPORT_INFO __nvoc_export_info_OBJREFCNT;

static const struct NVOC_RTTI __nvoc_rtti_OBJREFCNT_OBJREFCNT = {
    /*pClassDef=*/          &__nvoc_class_def_OBJREFCNT,
    /*dtor=*/               (NVOC_DYNAMIC_DTOR) &__nvoc_dtor_OBJREFCNT,
    /*offset=*/             0,
};

static const struct NVOC_RTTI __nvoc_rtti_OBJREFCNT_Object = {
    /*pClassDef=*/          &__nvoc_class_def_Object,
    /*dtor=*/               &__nvoc_destructFromBase,
    /*offset=*/             NV_OFFSETOF(OBJREFCNT, __nvoc_base_Object),
};

static const struct NVOC_CASTINFO __nvoc_castinfo_OBJREFCNT = {
    /*numRelatives=*/       2,
    /*relatives=*/ {
        &__nvoc_rtti_OBJREFCNT_OBJREFCNT,
        &__nvoc_rtti_OBJREFCNT_Object,
    },
};

const struct NVOC_CLASS_DEF __nvoc_class_def_OBJREFCNT = 
{
    /*classInfo=*/ {
        /*size=*/               sizeof(OBJREFCNT),
        /*classId=*/            classId(OBJREFCNT),
        /*providerId=*/         &__nvoc_rtti_provider,
#if NV_PRINTF_STRINGS_ALLOWED
        /*name=*/               "OBJREFCNT",
#endif
    },
    /*objCreatefn=*/        (NVOC_DYNAMIC_OBJ_CREATE) &__nvoc_objCreateDynamic_OBJREFCNT,
    /*pCastInfo=*/          &__nvoc_castinfo_OBJREFCNT,
    /*pExportInfo=*/        &__nvoc_export_info_OBJREFCNT
};

const struct NVOC_EXPORT_INFO __nvoc_export_info_OBJREFCNT = 
{
    /*numEntries=*/     0,
    /*pExportEntries=*/  0
};

void __nvoc_dtor_Object(Object*);
void __nvoc_dtor_OBJREFCNT(OBJREFCNT *pThis) {
    __nvoc_refcntDestruct(pThis);
    __nvoc_dtor_Object(&pThis->__nvoc_base_Object);
    PORT_UNREFERENCED_VARIABLE(pThis);
}

void __nvoc_init_dataField_OBJREFCNT(OBJREFCNT *pThis) {
    PORT_UNREFERENCED_VARIABLE(pThis);
}

NV_STATUS __nvoc_ctor_Object(Object* );
NV_STATUS __nvoc_ctor_OBJREFCNT(OBJREFCNT *pThis, Dynamic * arg_pParent, NvU32 arg_tag, RefcntStateChangeCallback * arg_pStateChangeCallback, RefcntResetCallback * arg_pResetCallback) {
    NV_STATUS status = NV_OK;
    status = __nvoc_ctor_Object(&pThis->__nvoc_base_Object);
    if (status != NV_OK) goto __nvoc_ctor_OBJREFCNT_fail_Object;
    __nvoc_init_dataField_OBJREFCNT(pThis);

    status = __nvoc_refcntConstruct(pThis, arg_pParent, arg_tag, arg_pStateChangeCallback, arg_pResetCallback);
    if (status != NV_OK) goto __nvoc_ctor_OBJREFCNT_fail__init;
    goto __nvoc_ctor_OBJREFCNT_exit; // Success

__nvoc_ctor_OBJREFCNT_fail__init:
    __nvoc_dtor_Object(&pThis->__nvoc_base_Object);
__nvoc_ctor_OBJREFCNT_fail_Object:
__nvoc_ctor_OBJREFCNT_exit:

    return status;
}

static void __nvoc_init_funcTable_OBJREFCNT_1(OBJREFCNT *pThis) {
    PORT_UNREFERENCED_VARIABLE(pThis);
}

void __nvoc_init_funcTable_OBJREFCNT(OBJREFCNT *pThis) {
    __nvoc_init_funcTable_OBJREFCNT_1(pThis);
}

void __nvoc_init_Object(Object*);
void __nvoc_init_OBJREFCNT(OBJREFCNT *pThis) {
    pThis->__nvoc_pbase_OBJREFCNT = pThis;
    pThis->__nvoc_pbase_Object = &pThis->__nvoc_base_Object;
    __nvoc_init_Object(&pThis->__nvoc_base_Object);
    __nvoc_init_funcTable_OBJREFCNT(pThis);
}

NV_STATUS __nvoc_objCreate_OBJREFCNT(OBJREFCNT **ppThis, Dynamic *pParent, NvU32 createFlags, Dynamic * arg_pParent, NvU32 arg_tag, RefcntStateChangeCallback * arg_pStateChangeCallback, RefcntResetCallback * arg_pResetCallback) {
    NV_STATUS status;
    Object *pParentObj;
    OBJREFCNT *pThis;

    pThis = portMemAllocNonPaged(sizeof(OBJREFCNT));
    if (pThis == NULL) return NV_ERR_NO_MEMORY;

    portMemSet(pThis, 0, sizeof(OBJREFCNT));

    __nvoc_initRtti(staticCast(pThis, Dynamic), &__nvoc_class_def_OBJREFCNT);

    if (pParent != NULL && !(createFlags & NVOC_OBJ_CREATE_FLAGS_PARENT_HALSPEC_ONLY))
    {
        pParentObj = dynamicCast(pParent, Object);
        objAddChild(pParentObj, &pThis->__nvoc_base_Object);
    }
    else
    {
        pThis->__nvoc_base_Object.pParent = NULL;
    }

    __nvoc_init_OBJREFCNT(pThis);
    status = __nvoc_ctor_OBJREFCNT(pThis, arg_pParent, arg_tag, arg_pStateChangeCallback, arg_pResetCallback);
    if (status != NV_OK) goto __nvoc_objCreate_OBJREFCNT_cleanup;

    *ppThis = pThis;
    return NV_OK;

__nvoc_objCreate_OBJREFCNT_cleanup:
    // do not call destructors here since the constructor already called them
    portMemFree(pThis);
    return status;
}

NV_STATUS __nvoc_objCreateDynamic_OBJREFCNT(OBJREFCNT **ppThis, Dynamic *pParent, NvU32 createFlags, va_list args) {
    NV_STATUS status;
    Dynamic * arg_pParent = va_arg(args, Dynamic *);
    NvU32 arg_tag = va_arg(args, NvU32);
    RefcntStateChangeCallback * arg_pStateChangeCallback = va_arg(args, RefcntStateChangeCallback *);
    RefcntResetCallback * arg_pResetCallback = va_arg(args, RefcntResetCallback *);

    status = __nvoc_objCreate_OBJREFCNT(ppThis, pParent, createFlags, arg_pParent, arg_tag, arg_pStateChangeCallback, arg_pResetCallback);

    return status;
}

