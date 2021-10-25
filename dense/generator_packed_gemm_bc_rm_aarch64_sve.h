/******************************************************************************
* Copyright (c) Friedrich Schiller University Jena - All rights reserved.     *
*               Xinyi Cui - All rights reserved.                              *
*               Adrian Krauss - All rights reserved.                          *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Breuer (Univ. Jena), Xinyi Cui, Adrian Krauss
******************************************************************************/

#ifndef GENERATOR_PACKED_GEMM_BC_RM_AARCH64_SVE_H
#define GENERATOR_PACKED_GEMM_BC_RM_AARCH64_SVE_H

#include <libxsmm_generator.h>
#include "generator_common.h"

LIBXSMM_API_INTERN void libxsmm_generator_packed_gemm_bc_rm_aarch64_sve( libxsmm_generated_code*         io_generated_code,
                                                                         const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                                                         const unsigned int              i_packed_width );

LIBXSMM_API_INTERN void libxsmm_generator_packed_gemm_bc_rm_aarch64_kloop_sve( libxsmm_generated_code*            io_generated_code,
                                                                               libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                                                               const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                                               const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                                               const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                                               const unsigned int                 i_packed_width,
                                                                               const unsigned int                 i_n_blocking );

LIBXSMM_API_INTERN void libxsmm_generator_packed_gemm_bc_rm_aarch64_kloop_sve_packed(  libxsmm_generated_code*            io_generated_code,
                                                                                       libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                                                                       const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                                                       const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                                                       const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                                                       const unsigned int                 i_packed_width,
                                                                                       const unsigned int                 i_sve_packed_remainder,
                                                                                       const unsigned int                 i_n_blocking, 
                                                                                       const unsigned char                i_pred_reg );

#endif /* GENERATOR_PACKED_GEMM_BC_RM_AARCH64_SVE_H */

