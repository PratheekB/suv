; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt -S -passes=instcombine %s |FileCheck %s

define <2 x i16> @patatino() {
; CHECK-LABEL: @patatino(
; CHECK-NEXT:    ret <2 x i16> zeroinitializer
;
  %tmp2 = getelementptr inbounds [1 x i16], [1 x i16]* null, i16 0, <2 x i16> undef
  %tmp3 = ptrtoint <2 x i16*> %tmp2 to <2 x i16>
  ret <2 x i16> %tmp3
}