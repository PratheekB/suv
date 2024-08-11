#define NVOC_PLATFORM_H_PRIVATE_ACCESS_ALLOWED
#include "nvoc/runtime.h"
#include "nvoc/rtti.h"
#include "nvtypes.h"
#include "nvport/nvport.h"
#include "nvport/inline/util_valist.h"
#include "utils/nvassert.h"
#include "g_platform_nvoc.h"

#ifdef DEBUG
char __nvoc_class_id_uniqueness_check_0xb543ae = 1;
#endif

extern const struct NVOC_CLASS_DEF __nvoc_class_def_OBJPFM;

extern const struct NVOC_CLASS_DEF __nvoc_class_def_Object;

void __nvoc_init_OBJPFM(OBJPFM*);
void __nvoc_init_funcTable_OBJPFM(OBJPFM*);
NV_STATUS __nvoc_ctor_OBJPFM(OBJPFM*);
void __nvoc_init_dataField_OBJPFM(OBJPFM*);
void __nvoc_dtor_OBJPFM(OBJPFM*);
extern const struct NVOC_EXPORT_INFO __nvoc_export_info_OBJPFM;

static const struct NVOC_RTTI __nvoc_rtti_OBJPFM_OBJPFM = {
    /*pClassDef=*/          &__nvoc_class_def_OBJPFM,
    /*dtor=*/               (NVOC_DYNAMIC_DTOR) &__nvoc_dtor_OBJPFM,
    /*offset=*/             0,
};

static const struct NVOC_RTTI __nvoc_rtti_OBJPFM_Object = {
    /*pClassDef=*/          &__nvoc_class_def_Object,
    /*dtor=*/               &__nvoc_destructFromBase,
    /*offset=*/             NV_OFFSETOF(OBJPFM, __nvoc_base_Object),
};

static const struct NVOC_CASTINFO __nvoc_castinfo_OBJPFM = {
    /*numRelatives=*/       2,
    /*relatives=*/ {
        &__nvoc_rtti_OBJPFM_OBJPFM,
        &__nvoc_rtti_OBJPFM_Object,
    },
};

const struct NVOC_CLASS_DEF __nvoc_class_def_OBJPFM = 
{
    /*classInfo=*/ {
        /*size=*/               sizeof(OBJPFM),
        /*classId=*/            classId(OBJPFM),
        /*providerId=*/         &__nvoc_rtti_provider,
#if NV_PRINTF_STRINGS_ALLOWED
        /*name=*/               "OBJPFM",
#endif
    },
    /*objCreatefn=*/        (NVOC_DYNAMIC_OBJ_CREATE) &__nvoc_objCreateDynamic_OBJPFM,
    /*pCastInfo=*/          &__nvoc_castinfo_OBJPFM,
    /*pExportInfo=*/        &__nvoc_export_info_OBJPFM
};

const struct NVOC_EXPORT_INFO __nvoc_export_info_OBJPFM = 
{
    /*numEntries=*/     0,
    /*pExportEntries=*/  0
};

void __nvoc_dtor_Object(Object*);
void __nvoc_dtor_OBJPFM(OBJPFM *pThis) {
    __nvoc_dtor_Object(&pThis->__nvoc_base_Object);
    PORT_UNREFERENCED_VARIABLE(pThis);
}

void __nvoc_init_dataField_OBJPFM(OBJPFM *pThis) {
    PORT_UNREFERENCED_VARIABLE(pThis);
    pThis->setProperty(pThis, PDB_PROP_PFM_SUPPORTS_ACPI, (0));
    pThis->setProperty(pThis, PDB_PROP_PFM_POSSIBLE_HIGHRES_BOOT, (0));
}

NV_STATUS __nvoc_ctor_Object(Object* );
NV_STATUS __nvoc_ctor_OBJPFM(OBJPFM *pThis) {
    NV_STATUS status = NV_OK;
    status = __nvoc_ctor_Object(&pThis->__nvoc_base_Object);
    if (status != NV_OK) goto __nvoc_ctor_OBJPFM_fail_Object;
    __nvoc_init_dataField_OBJPFM(pThis);

    status = __nvoc_pfmConstruct(pThis);
    if (status != NV_OK) goto __nvoc_ctor_OBJPFM_fail__init;
    goto __nvoc_ctor_OBJPFM_exit; // Success

__nvoc_ctor_OBJPFM_fail__init:
    __nvoc_dtor_Object(&pThis->__nvoc_base_Object);
__nvoc_ctor_OBJPFM_fail_Object:
__nvoc_ctor_OBJPFM_exit:

    return status;
}

static void __nvoc_init_funcTable_OBJPFM_1(OBJPFM *pThis) {
    PORT_UNREFERENCED_VARIABLE(pThis);
}

void __nvoc_init_funcTable_OBJPFM(OBJPFM *pThis) {
    __nvoc_init_funcTable_OBJPFM_1(pThis);
}

void __nvoc_init_Object(Object*);
void __nvoc_init_OBJPFM(OBJPFM *pThis) {
    pThis->__nvoc_pbase_OBJPFM = pThis;
    pThis->__nvoc_pbase_Object = &pThis->__nvoc_base_Object;
    __nvoc_init_Object(&pThis->__nvoc_base_Object);
    __nvoc_init_funcTable_OBJPFM(pThis);
}

NV_STATUS __nvoc_objCreate_OBJPFM(OBJPFM **ppThis, Dynamic *pParent, NvU32 createFlags) {
    NV_STATUS status;
    Object *pParentObj;
    OBJPFM *pThis;

    pThis = portMemAllocNonPaged(sizeof(OBJPFM));
    if (pThis == NULL) return NV_ERR_NO_MEMORY;

    portMemSet(pThis, 0, sizeof(OBJPFM));

    __nvoc_initRtti(staticCast(pThis, Dynamic), &__nvoc_class_def_OBJPFM);

    if (pParent != NULL && !(createFlags & NVOC_OBJ_CREATE_FLAGS_PARENT_HALSPEC_ONLY))
    {
        pParentObj = dynamicCast(pParent, Object);
        objAddChild(pParentObj, &pThis->__nvoc_base_Object);
    }
    else
    {
        pThis->__nvoc_base_Object.pParent = NULL;
    }

    __nvoc_init_OBJPFM(pThis);
    status = __nvoc_ctor_OBJPFM(pThis);
    if (status != NV_OK) goto __nvoc_objCreate_OBJPFM_cleanup;

    *ppThis = pThis;
    return NV_OK;

__nvoc_objCreate_OBJPFM_cleanup:
    // do not call destructors here since the constructor already called them
    portMemFree(pThis);
    return status;
}

NV_STATUS __nvoc_objCreateDynamic_OBJPFM(OBJPFM **ppThis, Dynamic *pParent, NvU32 createFlags, va_list args) {
    NV_STATUS status;

    status = __nvoc_objCreate_OBJPFM(ppThis, pParent, createFlags);

    return status;
}

