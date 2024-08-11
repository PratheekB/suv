#define NVOC_FABRIC_H_PRIVATE_ACCESS_ALLOWED
#include "nvoc/runtime.h"
#include "nvoc/rtti.h"
#include "nvtypes.h"
#include "nvport/nvport.h"
#include "nvport/inline/util_valist.h"
#include "utils/nvassert.h"
#include "g_fabric_nvoc.h"

#ifdef DEBUG
char __nvoc_class_id_uniqueness_check_0x0ac791 = 1;
#endif

extern const struct NVOC_CLASS_DEF __nvoc_class_def_Fabric;

extern const struct NVOC_CLASS_DEF __nvoc_class_def_Object;

void __nvoc_init_Fabric(Fabric*);
void __nvoc_init_funcTable_Fabric(Fabric*);
NV_STATUS __nvoc_ctor_Fabric(Fabric*);
void __nvoc_init_dataField_Fabric(Fabric*);
void __nvoc_dtor_Fabric(Fabric*);
extern const struct NVOC_EXPORT_INFO __nvoc_export_info_Fabric;

static const struct NVOC_RTTI __nvoc_rtti_Fabric_Fabric = {
    /*pClassDef=*/          &__nvoc_class_def_Fabric,
    /*dtor=*/               (NVOC_DYNAMIC_DTOR) &__nvoc_dtor_Fabric,
    /*offset=*/             0,
};

static const struct NVOC_RTTI __nvoc_rtti_Fabric_Object = {
    /*pClassDef=*/          &__nvoc_class_def_Object,
    /*dtor=*/               &__nvoc_destructFromBase,
    /*offset=*/             NV_OFFSETOF(Fabric, __nvoc_base_Object),
};

static const struct NVOC_CASTINFO __nvoc_castinfo_Fabric = {
    /*numRelatives=*/       2,
    /*relatives=*/ {
        &__nvoc_rtti_Fabric_Fabric,
        &__nvoc_rtti_Fabric_Object,
    },
};

const struct NVOC_CLASS_DEF __nvoc_class_def_Fabric = 
{
    /*classInfo=*/ {
        /*size=*/               sizeof(Fabric),
        /*classId=*/            classId(Fabric),
        /*providerId=*/         &__nvoc_rtti_provider,
#if NV_PRINTF_STRINGS_ALLOWED
        /*name=*/               "Fabric",
#endif
    },
    /*objCreatefn=*/        (NVOC_DYNAMIC_OBJ_CREATE) &__nvoc_objCreateDynamic_Fabric,
    /*pCastInfo=*/          &__nvoc_castinfo_Fabric,
    /*pExportInfo=*/        &__nvoc_export_info_Fabric
};

const struct NVOC_EXPORT_INFO __nvoc_export_info_Fabric = 
{
    /*numEntries=*/     0,
    /*pExportEntries=*/  0
};

void __nvoc_dtor_Object(Object*);
void __nvoc_dtor_Fabric(Fabric *pThis) {
    __nvoc_fabricDestruct(pThis);
    __nvoc_dtor_Object(&pThis->__nvoc_base_Object);
    PORT_UNREFERENCED_VARIABLE(pThis);
}

void __nvoc_init_dataField_Fabric(Fabric *pThis) {
    PORT_UNREFERENCED_VARIABLE(pThis);
}

NV_STATUS __nvoc_ctor_Object(Object* );
NV_STATUS __nvoc_ctor_Fabric(Fabric *pThis) {
    NV_STATUS status = NV_OK;
    status = __nvoc_ctor_Object(&pThis->__nvoc_base_Object);
    if (status != NV_OK) goto __nvoc_ctor_Fabric_fail_Object;
    __nvoc_init_dataField_Fabric(pThis);

    status = __nvoc_fabricConstruct(pThis);
    if (status != NV_OK) goto __nvoc_ctor_Fabric_fail__init;
    goto __nvoc_ctor_Fabric_exit; // Success

__nvoc_ctor_Fabric_fail__init:
    __nvoc_dtor_Object(&pThis->__nvoc_base_Object);
__nvoc_ctor_Fabric_fail_Object:
__nvoc_ctor_Fabric_exit:

    return status;
}

static void __nvoc_init_funcTable_Fabric_1(Fabric *pThis) {
    PORT_UNREFERENCED_VARIABLE(pThis);
}

void __nvoc_init_funcTable_Fabric(Fabric *pThis) {
    __nvoc_init_funcTable_Fabric_1(pThis);
}

void __nvoc_init_Object(Object*);
void __nvoc_init_Fabric(Fabric *pThis) {
    pThis->__nvoc_pbase_Fabric = pThis;
    pThis->__nvoc_pbase_Object = &pThis->__nvoc_base_Object;
    __nvoc_init_Object(&pThis->__nvoc_base_Object);
    __nvoc_init_funcTable_Fabric(pThis);
}

NV_STATUS __nvoc_objCreate_Fabric(Fabric **ppThis, Dynamic *pParent, NvU32 createFlags) {
    NV_STATUS status;
    Object *pParentObj;
    Fabric *pThis;

    pThis = portMemAllocNonPaged(sizeof(Fabric));
    if (pThis == NULL) return NV_ERR_NO_MEMORY;

    portMemSet(pThis, 0, sizeof(Fabric));

    __nvoc_initRtti(staticCast(pThis, Dynamic), &__nvoc_class_def_Fabric);

    if (pParent != NULL && !(createFlags & NVOC_OBJ_CREATE_FLAGS_PARENT_HALSPEC_ONLY))
    {
        pParentObj = dynamicCast(pParent, Object);
        objAddChild(pParentObj, &pThis->__nvoc_base_Object);
    }
    else
    {
        pThis->__nvoc_base_Object.pParent = NULL;
    }

    __nvoc_init_Fabric(pThis);
    status = __nvoc_ctor_Fabric(pThis);
    if (status != NV_OK) goto __nvoc_objCreate_Fabric_cleanup;

    *ppThis = pThis;
    return NV_OK;

__nvoc_objCreate_Fabric_cleanup:
    // do not call destructors here since the constructor already called them
    portMemFree(pThis);
    return status;
}

NV_STATUS __nvoc_objCreateDynamic_Fabric(Fabric **ppThis, Dynamic *pParent, NvU32 createFlags, va_list args) {
    NV_STATUS status;

    status = __nvoc_objCreate_Fabric(ppThis, pParent, createFlags);

    return status;
}

