; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py UTC_ARGS: --no_x86_scrub_mem_shuffle
; RUN: llc < %s -mtriple=x86_64-unknown-unknown | FileCheck %s --check-prefix=SSE
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+sse3 | FileCheck %s --check-prefix=SSE
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+ssse3 | FileCheck %s --check-prefix=SSE
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+sse4.1 | FileCheck %s --check-prefix=SSE
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+avx | FileCheck %s --check-prefix=AVX1
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+avx2 | FileCheck %s --check-prefix=AVX2

; PR21281
define <64 x i16> @interleave8x8(<8 x i16> %a, <8 x i16> %b, <8 x i16> %c, <8 x i16> %d, <8 x i16> %e, <8 x i16> %f, <8 x i16> %h, <8 x i16> %g) {
; SSE-LABEL: interleave8x8:
; SSE:       # %bb.0:
; SSE-NEXT:    movq %rdi, %rax
; SSE-NEXT:    movdqa %xmm0, %xmm8
; SSE-NEXT:    punpcklwd {{.*#+}} xmm8 = xmm8[0],xmm1[0],xmm8[1],xmm1[1],xmm8[2],xmm1[2],xmm8[3],xmm1[3]
; SSE-NEXT:    punpckhwd {{.*#+}} xmm0 = xmm0[4],xmm1[4],xmm0[5],xmm1[5],xmm0[6],xmm1[6],xmm0[7],xmm1[7]
; SSE-NEXT:    movdqa %xmm2, %xmm1
; SSE-NEXT:    punpcklwd {{.*#+}} xmm1 = xmm1[0],xmm3[0],xmm1[1],xmm3[1],xmm1[2],xmm3[2],xmm1[3],xmm3[3]
; SSE-NEXT:    punpckhwd {{.*#+}} xmm2 = xmm2[4],xmm3[4],xmm2[5],xmm3[5],xmm2[6],xmm3[6],xmm2[7],xmm3[7]
; SSE-NEXT:    movdqa %xmm0, %xmm3
; SSE-NEXT:    punpckhdq {{.*#+}} xmm3 = xmm3[2],xmm2[2],xmm3[3],xmm2[3]
; SSE-NEXT:    punpckldq {{.*#+}} xmm0 = xmm0[0],xmm2[0],xmm0[1],xmm2[1]
; SSE-NEXT:    movdqa %xmm8, %xmm2
; SSE-NEXT:    punpckhdq {{.*#+}} xmm2 = xmm2[2],xmm1[2],xmm2[3],xmm1[3]
; SSE-NEXT:    punpckldq {{.*#+}} xmm8 = xmm8[0],xmm1[0],xmm8[1],xmm1[1]
; SSE-NEXT:    movdqa %xmm4, %xmm1
; SSE-NEXT:    punpcklwd {{.*#+}} xmm1 = xmm1[0],xmm5[0],xmm1[1],xmm5[1],xmm1[2],xmm5[2],xmm1[3],xmm5[3]
; SSE-NEXT:    punpckhwd {{.*#+}} xmm4 = xmm4[4],xmm5[4],xmm4[5],xmm5[5],xmm4[6],xmm5[6],xmm4[7],xmm5[7]
; SSE-NEXT:    movdqa %xmm7, %xmm5
; SSE-NEXT:    punpcklwd {{.*#+}} xmm5 = xmm5[0],xmm6[0],xmm5[1],xmm6[1],xmm5[2],xmm6[2],xmm5[3],xmm6[3]
; SSE-NEXT:    punpckhwd {{.*#+}} xmm7 = xmm7[4],xmm6[4],xmm7[5],xmm6[5],xmm7[6],xmm6[6],xmm7[7],xmm6[7]
; SSE-NEXT:    movdqa %xmm4, %xmm6
; SSE-NEXT:    punpckhdq {{.*#+}} xmm6 = xmm6[2],xmm7[2],xmm6[3],xmm7[3]
; SSE-NEXT:    punpckldq {{.*#+}} xmm4 = xmm4[0],xmm7[0],xmm4[1],xmm7[1]
; SSE-NEXT:    movdqa %xmm1, %xmm7
; SSE-NEXT:    punpckhdq {{.*#+}} xmm7 = xmm7[2],xmm5[2],xmm7[3],xmm5[3]
; SSE-NEXT:    punpckldq {{.*#+}} xmm1 = xmm1[0],xmm5[0],xmm1[1],xmm5[1]
; SSE-NEXT:    movdqa %xmm8, %xmm5
; SSE-NEXT:    punpcklwd {{.*#+}} xmm5 = xmm5[0],xmm1[0],xmm5[1],xmm1[1],xmm5[2],xmm1[2],xmm5[3],xmm1[3]
; SSE-NEXT:    punpckhwd {{.*#+}} xmm8 = xmm8[4],xmm1[4],xmm8[5],xmm1[5],xmm8[6],xmm1[6],xmm8[7],xmm1[7]
; SSE-NEXT:    movdqa %xmm2, %xmm1
; SSE-NEXT:    punpcklwd {{.*#+}} xmm1 = xmm1[0],xmm7[0],xmm1[1],xmm7[1],xmm1[2],xmm7[2],xmm1[3],xmm7[3]
; SSE-NEXT:    punpckhwd {{.*#+}} xmm2 = xmm2[4],xmm7[4],xmm2[5],xmm7[5],xmm2[6],xmm7[6],xmm2[7],xmm7[7]
; SSE-NEXT:    movdqa %xmm0, %xmm7
; SSE-NEXT:    punpcklwd {{.*#+}} xmm7 = xmm7[0],xmm4[0],xmm7[1],xmm4[1],xmm7[2],xmm4[2],xmm7[3],xmm4[3]
; SSE-NEXT:    punpckhwd {{.*#+}} xmm0 = xmm0[4],xmm4[4],xmm0[5],xmm4[5],xmm0[6],xmm4[6],xmm0[7],xmm4[7]
; SSE-NEXT:    movdqa %xmm3, %xmm4
; SSE-NEXT:    punpcklwd {{.*#+}} xmm4 = xmm4[0],xmm6[0],xmm4[1],xmm6[1],xmm4[2],xmm6[2],xmm4[3],xmm6[3]
; SSE-NEXT:    punpckhwd {{.*#+}} xmm3 = xmm3[4],xmm6[4],xmm3[5],xmm6[5],xmm3[6],xmm6[6],xmm3[7],xmm6[7]
; SSE-NEXT:    movdqa %xmm3, 112(%rdi)
; SSE-NEXT:    movdqa %xmm4, 96(%rdi)
; SSE-NEXT:    movdqa %xmm0, 80(%rdi)
; SSE-NEXT:    movdqa %xmm7, 64(%rdi)
; SSE-NEXT:    movdqa %xmm2, 48(%rdi)
; SSE-NEXT:    movdqa %xmm1, 32(%rdi)
; SSE-NEXT:    movdqa %xmm8, 16(%rdi)
; SSE-NEXT:    movdqa %xmm5, (%rdi)
; SSE-NEXT:    retq
;
; AVX1-LABEL: interleave8x8:
; AVX1:       # %bb.0:
; AVX1-NEXT:    vpunpcklwd {{.*#+}} xmm8 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; AVX1-NEXT:    vpunpckhwd {{.*#+}} xmm0 = xmm0[4],xmm1[4],xmm0[5],xmm1[5],xmm0[6],xmm1[6],xmm0[7],xmm1[7]
; AVX1-NEXT:    vpunpcklwd {{.*#+}} xmm1 = xmm2[0],xmm3[0],xmm2[1],xmm3[1],xmm2[2],xmm3[2],xmm2[3],xmm3[3]
; AVX1-NEXT:    vpunpckhwd {{.*#+}} xmm2 = xmm2[4],xmm3[4],xmm2[5],xmm3[5],xmm2[6],xmm3[6],xmm2[7],xmm3[7]
; AVX1-NEXT:    vpunpckhdq {{.*#+}} xmm9 = xmm0[2],xmm2[2],xmm0[3],xmm2[3]
; AVX1-NEXT:    vpunpckldq {{.*#+}} xmm2 = xmm0[0],xmm2[0],xmm0[1],xmm2[1]
; AVX1-NEXT:    vpunpckhdq {{.*#+}} xmm3 = xmm8[2],xmm1[2],xmm8[3],xmm1[3]
; AVX1-NEXT:    vpunpckldq {{.*#+}} xmm0 = xmm8[0],xmm1[0],xmm8[1],xmm1[1]
; AVX1-NEXT:    vpunpcklwd {{.*#+}} xmm1 = xmm4[0],xmm5[0],xmm4[1],xmm5[1],xmm4[2],xmm5[2],xmm4[3],xmm5[3]
; AVX1-NEXT:    vpunpckhwd {{.*#+}} xmm4 = xmm4[4],xmm5[4],xmm4[5],xmm5[5],xmm4[6],xmm5[6],xmm4[7],xmm5[7]
; AVX1-NEXT:    vpunpcklwd {{.*#+}} xmm5 = xmm7[0],xmm6[0],xmm7[1],xmm6[1],xmm7[2],xmm6[2],xmm7[3],xmm6[3]
; AVX1-NEXT:    vpunpckhwd {{.*#+}} xmm6 = xmm7[4],xmm6[4],xmm7[5],xmm6[5],xmm7[6],xmm6[6],xmm7[7],xmm6[7]
; AVX1-NEXT:    vpunpckhdq {{.*#+}} xmm7 = xmm4[2],xmm6[2],xmm4[3],xmm6[3]
; AVX1-NEXT:    vpunpckldq {{.*#+}} xmm4 = xmm4[0],xmm6[0],xmm4[1],xmm6[1]
; AVX1-NEXT:    vpunpckhdq {{.*#+}} xmm6 = xmm1[2],xmm5[2],xmm1[3],xmm5[3]
; AVX1-NEXT:    vpunpckldq {{.*#+}} xmm1 = xmm1[0],xmm5[0],xmm1[1],xmm5[1]
; AVX1-NEXT:    vpunpckhwd {{.*#+}} xmm5 = xmm0[4],xmm1[4],xmm0[5],xmm1[5],xmm0[6],xmm1[6],xmm0[7],xmm1[7]
; AVX1-NEXT:    vpunpcklwd {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; AVX1-NEXT:    vinsertf128 $1, %xmm5, %ymm0, %ymm0
; AVX1-NEXT:    vpunpckhwd {{.*#+}} xmm1 = xmm3[4],xmm6[4],xmm3[5],xmm6[5],xmm3[6],xmm6[6],xmm3[7],xmm6[7]
; AVX1-NEXT:    vpunpcklwd {{.*#+}} xmm3 = xmm3[0],xmm6[0],xmm3[1],xmm6[1],xmm3[2],xmm6[2],xmm3[3],xmm6[3]
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm3, %ymm1
; AVX1-NEXT:    vpunpckhwd {{.*#+}} xmm3 = xmm2[4],xmm4[4],xmm2[5],xmm4[5],xmm2[6],xmm4[6],xmm2[7],xmm4[7]
; AVX1-NEXT:    vpunpcklwd {{.*#+}} xmm2 = xmm2[0],xmm4[0],xmm2[1],xmm4[1],xmm2[2],xmm4[2],xmm2[3],xmm4[3]
; AVX1-NEXT:    vinsertf128 $1, %xmm3, %ymm2, %ymm2
; AVX1-NEXT:    vpunpckhwd {{.*#+}} xmm3 = xmm9[4],xmm7[4],xmm9[5],xmm7[5],xmm9[6],xmm7[6],xmm9[7],xmm7[7]
; AVX1-NEXT:    vpunpcklwd {{.*#+}} xmm4 = xmm9[0],xmm7[0],xmm9[1],xmm7[1],xmm9[2],xmm7[2],xmm9[3],xmm7[3]
; AVX1-NEXT:    vinsertf128 $1, %xmm3, %ymm4, %ymm3
; AVX1-NEXT:    retq
;
; AVX2-LABEL: interleave8x8:
; AVX2:       # %bb.0:
; AVX2-NEXT:    vpunpcklwd {{.*#+}} xmm8 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; AVX2-NEXT:    vpunpckhwd {{.*#+}} xmm0 = xmm0[4],xmm1[4],xmm0[5],xmm1[5],xmm0[6],xmm1[6],xmm0[7],xmm1[7]
; AVX2-NEXT:    vpunpcklwd {{.*#+}} xmm1 = xmm2[0],xmm3[0],xmm2[1],xmm3[1],xmm2[2],xmm3[2],xmm2[3],xmm3[3]
; AVX2-NEXT:    vpunpckhwd {{.*#+}} xmm2 = xmm2[4],xmm3[4],xmm2[5],xmm3[5],xmm2[6],xmm3[6],xmm2[7],xmm3[7]
; AVX2-NEXT:    vpunpckhdq {{.*#+}} xmm9 = xmm0[2],xmm2[2],xmm0[3],xmm2[3]
; AVX2-NEXT:    vpunpckldq {{.*#+}} xmm2 = xmm0[0],xmm2[0],xmm0[1],xmm2[1]
; AVX2-NEXT:    vpunpckhdq {{.*#+}} xmm3 = xmm8[2],xmm1[2],xmm8[3],xmm1[3]
; AVX2-NEXT:    vpunpckldq {{.*#+}} xmm0 = xmm8[0],xmm1[0],xmm8[1],xmm1[1]
; AVX2-NEXT:    vpunpcklwd {{.*#+}} xmm1 = xmm4[0],xmm5[0],xmm4[1],xmm5[1],xmm4[2],xmm5[2],xmm4[3],xmm5[3]
; AVX2-NEXT:    vpunpckhwd {{.*#+}} xmm4 = xmm4[4],xmm5[4],xmm4[5],xmm5[5],xmm4[6],xmm5[6],xmm4[7],xmm5[7]
; AVX2-NEXT:    vpunpcklwd {{.*#+}} xmm5 = xmm7[0],xmm6[0],xmm7[1],xmm6[1],xmm7[2],xmm6[2],xmm7[3],xmm6[3]
; AVX2-NEXT:    vpunpckhwd {{.*#+}} xmm6 = xmm7[4],xmm6[4],xmm7[5],xmm6[5],xmm7[6],xmm6[6],xmm7[7],xmm6[7]
; AVX2-NEXT:    vpunpckhdq {{.*#+}} xmm7 = xmm4[2],xmm6[2],xmm4[3],xmm6[3]
; AVX2-NEXT:    vpunpckldq {{.*#+}} xmm4 = xmm4[0],xmm6[0],xmm4[1],xmm6[1]
; AVX2-NEXT:    vpunpckhdq {{.*#+}} xmm6 = xmm1[2],xmm5[2],xmm1[3],xmm5[3]
; AVX2-NEXT:    vpunpckldq {{.*#+}} xmm1 = xmm1[0],xmm5[0],xmm1[1],xmm5[1]
; AVX2-NEXT:    vpunpckhwd {{.*#+}} xmm5 = xmm0[4],xmm1[4],xmm0[5],xmm1[5],xmm0[6],xmm1[6],xmm0[7],xmm1[7]
; AVX2-NEXT:    vpunpcklwd {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; AVX2-NEXT:    vinserti128 $1, %xmm5, %ymm0, %ymm0
; AVX2-NEXT:    vpunpckhwd {{.*#+}} xmm1 = xmm3[4],xmm6[4],xmm3[5],xmm6[5],xmm3[6],xmm6[6],xmm3[7],xmm6[7]
; AVX2-NEXT:    vpunpcklwd {{.*#+}} xmm3 = xmm3[0],xmm6[0],xmm3[1],xmm6[1],xmm3[2],xmm6[2],xmm3[3],xmm6[3]
; AVX2-NEXT:    vinserti128 $1, %xmm1, %ymm3, %ymm1
; AVX2-NEXT:    vpunpckhwd {{.*#+}} xmm3 = xmm2[4],xmm4[4],xmm2[5],xmm4[5],xmm2[6],xmm4[6],xmm2[7],xmm4[7]
; AVX2-NEXT:    vpunpcklwd {{.*#+}} xmm2 = xmm2[0],xmm4[0],xmm2[1],xmm4[1],xmm2[2],xmm4[2],xmm2[3],xmm4[3]
; AVX2-NEXT:    vinserti128 $1, %xmm3, %ymm2, %ymm2
; AVX2-NEXT:    vpunpckhwd {{.*#+}} xmm3 = xmm9[4],xmm7[4],xmm9[5],xmm7[5],xmm9[6],xmm7[6],xmm9[7],xmm7[7]
; AVX2-NEXT:    vpunpcklwd {{.*#+}} xmm4 = xmm9[0],xmm7[0],xmm9[1],xmm7[1],xmm9[2],xmm7[2],xmm9[3],xmm7[3]
; AVX2-NEXT:    vinserti128 $1, %xmm3, %ymm4, %ymm3
; AVX2-NEXT:    retq
  %ab = shufflevector <8 x i16> %a, <8 x i16> %b, <16 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11, i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
  %cd = shufflevector <8 x i16> %c, <8 x i16> %d, <16 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11, i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
  %ab32 = bitcast <16 x i16> %ab to <8 x i32>
  %cd32 = bitcast <16 x i16> %cd to <8 x i32>
  %abcd32 = shufflevector <8 x i32> %ab32, <8 x i32> %cd32, <16 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11, i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
  %abcd = bitcast <16 x i32> %abcd32 to <32 x i16>

  %ef = shufflevector <8 x i16> %e, <8 x i16> %f, <16 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11, i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
  %gh = shufflevector <8 x i16> %g, <8 x i16> %h, <16 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11, i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
  %ef32 = bitcast <16 x i16> %ef to <8 x i32>
  %gh32 = bitcast <16 x i16> %gh to <8 x i32>
  %efgh32 = shufflevector <8 x i32> %ef32, <8 x i32> %gh32, <16 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11, i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
  %efgh = bitcast <16 x i32> %efgh32 to <32 x i16>

  %result = shufflevector <32 x i16> %abcd, <32 x i16> %efgh, <64 x i32> <i32 0, i32 32, i32 1, i32 33, i32 2, i32 34, i32 3, i32 35, i32 4, i32 36, i32 5, i32 37, i32 6, i32 38, i32 7, i32 39, i32 8, i32 40, i32 9, i32 41, i32 10, i32 42, i32 11, i32 43, i32 12, i32 44, i32 13, i32 45, i32 14, i32 46, i32 15, i32 47, i32 16, i32 48, i32 17, i32 49, i32 18, i32 50, i32 19, i32 51, i32 20, i32 52, i32 21, i32 53, i32 22, i32 54, i32 23, i32 55, i32 24, i32 56, i32 25, i32 57, i32 26, i32 58, i32 27, i32 59, i32 28, i32 60, i32 29, i32 61, i32 30, i32 62, i32 31, i32 63>
  ret <64 x i16> %result
}

define void @splat2_i8(ptr %s, ptr %d) {
; SSE-LABEL: splat2_i8:
; SSE:       # %bb.0:
; SSE-NEXT:    movdqu (%rdi), %xmm0
; SSE-NEXT:    movdqu 16(%rdi), %xmm1
; SSE-NEXT:    movdqa %xmm0, %xmm2
; SSE-NEXT:    punpcklbw {{.*#+}} xmm2 = xmm2[0],xmm0[0],xmm2[1],xmm0[1],xmm2[2],xmm0[2],xmm2[3],xmm0[3],xmm2[4],xmm0[4],xmm2[5],xmm0[5],xmm2[6],xmm0[6],xmm2[7],xmm0[7]
; SSE-NEXT:    punpckhbw {{.*#+}} xmm0 = xmm0[8,8,9,9,10,10,11,11,12,12,13,13,14,14,15,15]
; SSE-NEXT:    movdqa %xmm1, %xmm3
; SSE-NEXT:    punpcklbw {{.*#+}} xmm3 = xmm3[0],xmm1[0],xmm3[1],xmm1[1],xmm3[2],xmm1[2],xmm3[3],xmm1[3],xmm3[4],xmm1[4],xmm3[5],xmm1[5],xmm3[6],xmm1[6],xmm3[7],xmm1[7]
; SSE-NEXT:    punpckhbw {{.*#+}} xmm1 = xmm1[8,8,9,9,10,10,11,11,12,12,13,13,14,14,15,15]
; SSE-NEXT:    movdqu %xmm1, 48(%rsi)
; SSE-NEXT:    movdqu %xmm3, 32(%rsi)
; SSE-NEXT:    movdqu %xmm0, 16(%rsi)
; SSE-NEXT:    movdqu %xmm2, (%rsi)
; SSE-NEXT:    retq
;
; AVX1-LABEL: splat2_i8:
; AVX1:       # %bb.0:
; AVX1-NEXT:    vmovdqu (%rdi), %xmm0
; AVX1-NEXT:    vmovdqu 16(%rdi), %xmm1
; AVX1-NEXT:    vpunpcklbw {{.*#+}} xmm2 = xmm0[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
; AVX1-NEXT:    vpunpckhbw {{.*#+}} xmm0 = xmm0[8,8,9,9,10,10,11,11,12,12,13,13,14,14,15,15]
; AVX1-NEXT:    vpunpcklbw {{.*#+}} xmm3 = xmm1[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
; AVX1-NEXT:    vpunpckhbw {{.*#+}} xmm1 = xmm1[8,8,9,9,10,10,11,11,12,12,13,13,14,14,15,15]
; AVX1-NEXT:    vmovdqu %xmm1, 48(%rsi)
; AVX1-NEXT:    vmovdqu %xmm3, 32(%rsi)
; AVX1-NEXT:    vmovdqu %xmm0, 16(%rsi)
; AVX1-NEXT:    vmovdqu %xmm2, (%rsi)
; AVX1-NEXT:    retq
;
; AVX2-LABEL: splat2_i8:
; AVX2:       # %bb.0:
; AVX2-NEXT:    vpermq $216, (%rdi), %ymm0 # ymm0 = mem[0,2,1,3]
; AVX2-NEXT:    vpunpcklbw {{.*#+}} ymm1 = ymm0[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,16,16,17,17,18,18,19,19,20,20,21,21,22,22,23,23]
; AVX2-NEXT:    vpunpckhbw {{.*#+}} ymm0 = ymm0[8,8,9,9,10,10,11,11,12,12,13,13,14,14,15,15,24,24,25,25,26,26,27,27,28,28,29,29,30,30,31,31]
; AVX2-NEXT:    vmovdqu %ymm0, 32(%rsi)
; AVX2-NEXT:    vmovdqu %ymm1, (%rsi)
; AVX2-NEXT:    vzeroupper
; AVX2-NEXT:    retq
  %ld32 = load <32 x i8>, ptr %s, align 1
  %cat = shufflevector <32 x i8> %ld32, <32 x i8> undef, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %cat2 = shufflevector <64 x i8> %cat, <64 x i8> undef, <64 x i32> <i32 0, i32 32, i32 1, i32 33, i32 2, i32 34, i32 3, i32 35, i32 4, i32 36, i32 5, i32 37, i32 6, i32 38, i32 7, i32 39, i32 8, i32 40, i32 9, i32 41, i32 10, i32 42, i32 11, i32 43, i32 12, i32 44, i32 13, i32 45, i32 14, i32 46, i32 15, i32 47, i32 16, i32 48, i32 17, i32 49, i32 18, i32 50, i32 19, i32 51, i32 20, i32 52, i32 21, i32 53, i32 22, i32 54, i32 23, i32 55, i32 24, i32 56, i32 25, i32 57, i32 26, i32 58, i32 27, i32 59, i32 28, i32 60, i32 29, i32 61, i32 30, i32 62, i32 31, i32 63>
  store <64 x i8> %cat2, ptr %d, align 1
  ret void
}

define void @splat2_i16(ptr %s, ptr %d) {
; SSE-LABEL: splat2_i16:
; SSE:       # %bb.0:
; SSE-NEXT:    movdqu (%rdi), %xmm0
; SSE-NEXT:    movdqu 16(%rdi), %xmm1
; SSE-NEXT:    movdqa %xmm0, %xmm2
; SSE-NEXT:    punpcklwd {{.*#+}} xmm2 = xmm2[0],xmm0[0],xmm2[1],xmm0[1],xmm2[2],xmm0[2],xmm2[3],xmm0[3]
; SSE-NEXT:    punpckhwd {{.*#+}} xmm0 = xmm0[4,4,5,5,6,6,7,7]
; SSE-NEXT:    movdqa %xmm1, %xmm3
; SSE-NEXT:    punpcklwd {{.*#+}} xmm3 = xmm3[0],xmm1[0],xmm3[1],xmm1[1],xmm3[2],xmm1[2],xmm3[3],xmm1[3]
; SSE-NEXT:    punpckhwd {{.*#+}} xmm1 = xmm1[4,4,5,5,6,6,7,7]
; SSE-NEXT:    movdqu %xmm1, 48(%rsi)
; SSE-NEXT:    movdqu %xmm3, 32(%rsi)
; SSE-NEXT:    movdqu %xmm0, 16(%rsi)
; SSE-NEXT:    movdqu %xmm2, (%rsi)
; SSE-NEXT:    retq
;
; AVX1-LABEL: splat2_i16:
; AVX1:       # %bb.0:
; AVX1-NEXT:    vmovdqu (%rdi), %xmm0
; AVX1-NEXT:    vmovdqu 16(%rdi), %xmm1
; AVX1-NEXT:    vpunpcklwd {{.*#+}} xmm2 = xmm0[0,0,1,1,2,2,3,3]
; AVX1-NEXT:    vpunpckhwd {{.*#+}} xmm0 = xmm0[4,4,5,5,6,6,7,7]
; AVX1-NEXT:    vpunpcklwd {{.*#+}} xmm3 = xmm1[0,0,1,1,2,2,3,3]
; AVX1-NEXT:    vpunpckhwd {{.*#+}} xmm1 = xmm1[4,4,5,5,6,6,7,7]
; AVX1-NEXT:    vmovdqu %xmm1, 48(%rsi)
; AVX1-NEXT:    vmovdqu %xmm3, 32(%rsi)
; AVX1-NEXT:    vmovdqu %xmm0, 16(%rsi)
; AVX1-NEXT:    vmovdqu %xmm2, (%rsi)
; AVX1-NEXT:    retq
;
; AVX2-LABEL: splat2_i16:
; AVX2:       # %bb.0:
; AVX2-NEXT:    vpermq $216, (%rdi), %ymm0 # ymm0 = mem[0,2,1,3]
; AVX2-NEXT:    vpunpcklwd {{.*#+}} ymm1 = ymm0[0,0,1,1,2,2,3,3,8,8,9,9,10,10,11,11]
; AVX2-NEXT:    vpunpckhwd {{.*#+}} ymm0 = ymm0[4,4,5,5,6,6,7,7,12,12,13,13,14,14,15,15]
; AVX2-NEXT:    vmovdqu %ymm0, 32(%rsi)
; AVX2-NEXT:    vmovdqu %ymm1, (%rsi)
; AVX2-NEXT:    vzeroupper
; AVX2-NEXT:    retq
  %ld32 = load <16 x i16>, ptr %s, align 1
  %cat = shufflevector <16 x i16> %ld32, <16 x i16> undef, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %cat2 = shufflevector <32 x i16> %cat, <32 x i16> undef, <32 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23, i32 8, i32 24, i32 9, i32 25, i32 10, i32 26, i32 11, i32 27, i32 12, i32 28, i32 13, i32 29, i32 14, i32 30, i32 15, i32 31>
  store <32 x i16> %cat2, ptr %d, align 1
  ret void
}

define void @splat2_i32(ptr %s, ptr %d) {
; SSE-LABEL: splat2_i32:
; SSE:       # %bb.0:
; SSE-NEXT:    movdqu (%rdi), %xmm0
; SSE-NEXT:    movdqu 16(%rdi), %xmm1
; SSE-NEXT:    pshufd {{.*#+}} xmm2 = xmm0[0,0,1,1]
; SSE-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[2,2,3,3]
; SSE-NEXT:    pshufd {{.*#+}} xmm3 = xmm1[0,0,1,1]
; SSE-NEXT:    pshufd {{.*#+}} xmm1 = xmm1[2,2,3,3]
; SSE-NEXT:    movdqu %xmm1, 48(%rsi)
; SSE-NEXT:    movdqu %xmm3, 32(%rsi)
; SSE-NEXT:    movdqu %xmm0, 16(%rsi)
; SSE-NEXT:    movdqu %xmm2, (%rsi)
; SSE-NEXT:    retq
;
; AVX1-LABEL: splat2_i32:
; AVX1:       # %bb.0:
; AVX1-NEXT:    vmovups (%rdi), %xmm0
; AVX1-NEXT:    vmovups 16(%rdi), %xmm1
; AVX1-NEXT:    vpermilps {{.*#+}} xmm2 = xmm0[0,0,1,1]
; AVX1-NEXT:    vpermilps {{.*#+}} xmm0 = xmm0[2,2,3,3]
; AVX1-NEXT:    vpermilps {{.*#+}} xmm3 = xmm1[0,0,1,1]
; AVX1-NEXT:    vpermilps {{.*#+}} xmm1 = xmm1[2,2,3,3]
; AVX1-NEXT:    vmovups %xmm1, 48(%rsi)
; AVX1-NEXT:    vmovups %xmm3, 32(%rsi)
; AVX1-NEXT:    vmovups %xmm0, 16(%rsi)
; AVX1-NEXT:    vmovups %xmm2, (%rsi)
; AVX1-NEXT:    retq
;
; AVX2-LABEL: splat2_i32:
; AVX2:       # %bb.0:
; AVX2-NEXT:    vpermpd $216, (%rdi), %ymm0 # ymm0 = mem[0,2,1,3]
; AVX2-NEXT:    vpermilps {{.*#+}} ymm1 = ymm0[0,0,1,1,4,4,5,5]
; AVX2-NEXT:    vpermilps {{.*#+}} ymm0 = ymm0[2,2,3,3,6,6,7,7]
; AVX2-NEXT:    vmovups %ymm0, 32(%rsi)
; AVX2-NEXT:    vmovups %ymm1, (%rsi)
; AVX2-NEXT:    vzeroupper
; AVX2-NEXT:    retq
  %ld32 = load <8 x i32>, ptr %s, align 1
  %cat = shufflevector <8 x i32> %ld32, <8 x i32> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %cat2 = shufflevector <16 x i32> %cat, <16 x i32> undef, <16 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11, i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
  store <16 x i32> %cat2, ptr %d, align 1
  ret void
}

define void @splat2_i64(ptr %s, ptr %d) {
; SSE-LABEL: splat2_i64:
; SSE:       # %bb.0:
; SSE-NEXT:    movdqu (%rdi), %xmm0
; SSE-NEXT:    movdqu 16(%rdi), %xmm1
; SSE-NEXT:    pshufd {{.*#+}} xmm2 = xmm0[0,1,0,1]
; SSE-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[2,3,2,3]
; SSE-NEXT:    pshufd {{.*#+}} xmm3 = xmm1[0,1,0,1]
; SSE-NEXT:    pshufd {{.*#+}} xmm1 = xmm1[2,3,2,3]
; SSE-NEXT:    movdqu %xmm1, 48(%rsi)
; SSE-NEXT:    movdqu %xmm3, 32(%rsi)
; SSE-NEXT:    movdqu %xmm0, 16(%rsi)
; SSE-NEXT:    movdqu %xmm2, (%rsi)
; SSE-NEXT:    retq
;
; AVX1-LABEL: splat2_i64:
; AVX1:       # %bb.0:
; AVX1-NEXT:    vperm2f128 $51, (%rdi), %ymm0, %ymm0 # ymm0 = mem[2,3,2,3]
; AVX1-NEXT:    vpermilpd {{.*#+}} ymm0 = ymm0[0,0,3,3]
; AVX1-NEXT:    vbroadcastf128 (%rdi), %ymm1 # ymm1 = mem[0,1,0,1]
; AVX1-NEXT:    vpermilpd {{.*#+}} ymm1 = ymm1[0,0,3,3]
; AVX1-NEXT:    vmovupd %ymm0, 32(%rsi)
; AVX1-NEXT:    vmovupd %ymm1, (%rsi)
; AVX1-NEXT:    vzeroupper
; AVX1-NEXT:    retq
;
; AVX2-LABEL: splat2_i64:
; AVX2:       # %bb.0:
; AVX2-NEXT:    vmovups (%rdi), %ymm0
; AVX2-NEXT:    vpermpd {{.*#+}} ymm1 = ymm0[0,0,1,1]
; AVX2-NEXT:    vpermpd {{.*#+}} ymm0 = ymm0[2,2,3,3]
; AVX2-NEXT:    vmovups %ymm0, 32(%rsi)
; AVX2-NEXT:    vmovups %ymm1, (%rsi)
; AVX2-NEXT:    vzeroupper
; AVX2-NEXT:    retq
  %ld32 = load <4 x i64>, ptr %s, align 1
  %cat = shufflevector <4 x i64> %ld32, <4 x i64> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3>
  %cat2 = shufflevector <8 x i64> %cat, <8 x i64> undef, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  store <8 x i64> %cat2, ptr %d, align 1
  ret void
}