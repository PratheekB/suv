; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s
;
; Check that a widening truncate to a vector of i1 elements can be handled.

define void @pr32275(<4 x i8> %B15) {
; CHECK-LABEL: pr32275:
; CHECK:       # %bb.0: # %BB
; CHECK-NEXT:    vlgvb %r0, %v24, 3
; CHECK-NEXT:    vlvgp %v0, %r0, %r0
; CHECK-NEXT:    vrepif %v1, 1
; CHECK-NEXT:    vn %v0, %v0, %v1
; CHECK-NEXT:    vlgvf %r0, %v0, 3
; CHECK-NEXT:  .LBB0_1: # %CF34
; CHECK-NEXT:    # =>This Inner Loop Header: Depth=1
; CHECK-NEXT:    cijlh %r0, 0, .LBB0_1
; CHECK-NEXT:  # %bb.2: # %CF36
; CHECK-NEXT:    br %r14
BB:
  br label %CF34

CF34:
  %Tr24 = trunc <4 x i8> %B15 to <4 x i1>
  %E28 = extractelement <4 x i1> %Tr24, i32 3
  br i1 %E28, label %CF34, label %CF36

CF36:
  ret void
}