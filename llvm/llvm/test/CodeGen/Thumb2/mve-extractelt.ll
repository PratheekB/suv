; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple=thumbv8.1m.main-none-none-eabi -mattr=+mve -verify-machineinstrs %s -o - | FileCheck %s

define arm_aapcs_vfpcc i32 @u8_explicit_extend(<16 x i8> %a) {
; CHECK-LABEL: u8_explicit_extend:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    vmov.u8 r0, q0[10]
; CHECK-NEXT:    bx lr
entry:
  %0 = extractelement <16 x i8> %a, i32 10
  %1 = zext i8 %0 to i32
  ret i32 %1
}

define arm_aapcs_vfpcc i32 @s8_explicit_extend(<16 x i8> %a) {
; CHECK-LABEL: s8_explicit_extend:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    vmov.s8 r0, q0[10]
; CHECK-NEXT:    bx lr
entry:
  %0 = extractelement <16 x i8> %a, i32 10
  %1 = sext i8 %0 to i32
  ret i32 %1
}

define arm_aapcs_vfpcc i8 @u8_extend_via_pcs(<16 x i8> %a) {
; CHECK-LABEL: u8_extend_via_pcs:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    vmov.u8 r0, q0[10]
; CHECK-NEXT:    bx lr
entry:
  %0 = extractelement <16 x i8> %a, i32 10
  ret i8 %0
}

define arm_aapcs_vfpcc signext i8 @s8_extend_via_pcs(<16 x i8> %a) {
; CHECK-LABEL: s8_extend_via_pcs:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    vmov.s8 r0, q0[10]
; CHECK-NEXT:    bx lr
entry:
  %0 = extractelement <16 x i8> %a, i32 10
  ret i8 %0
}

define arm_aapcs_vfpcc i32 @u16_explicit_extend(<8 x i16> %a) {
; CHECK-LABEL: u16_explicit_extend:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    vmov.u16 r0, q0[5]
; CHECK-NEXT:    bx lr
entry:
  %0 = extractelement <8 x i16> %a, i32 5
  %1 = zext i16 %0 to i32
  ret i32 %1
}

define arm_aapcs_vfpcc i32 @s16_explicit_extend(<8 x i16> %a) {
; CHECK-LABEL: s16_explicit_extend:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    vmov.s16 r0, q0[5]
; CHECK-NEXT:    bx lr
entry:
  %0 = extractelement <8 x i16> %a, i32 5
  %1 = sext i16 %0 to i32
  ret i32 %1
}

define arm_aapcs_vfpcc i16 @u16_extend_via_pcs(<8 x i16> %a) {
; CHECK-LABEL: u16_extend_via_pcs:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    vmov.u16 r0, q0[5]
; CHECK-NEXT:    bx lr
entry:
  %0 = extractelement <8 x i16> %a, i32 5
  ret i16 %0
}

define arm_aapcs_vfpcc signext i16 @s16_extend_via_pcs(<8 x i16> %a) {
; CHECK-LABEL: s16_extend_via_pcs:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    vmov.s16 r0, q0[5]
; CHECK-NEXT:    bx lr
entry:
  %0 = extractelement <8 x i16> %a, i32 5
  ret i16 %0
}