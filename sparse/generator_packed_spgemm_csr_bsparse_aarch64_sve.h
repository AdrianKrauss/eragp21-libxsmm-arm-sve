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

#ifndef GENERATOR_PACKED_SPGEMM_CSR_BSPARSE_AARCH64_SVE_H
#define GENERATOR_PACKED_SPGEMM_CSR_BSPARSE_AARCH64_SVE_H

#include <libxsmm_generator.h>
#include "generator_common.h"
#include "generator_aarch64_instructions.h"

LIBXSMM_API_INTERN
void libxsmm_generator_packed_spgemm_csr_bsparse_aarch64_sve( libxsmm_generated_code*         io_generated_code,
                                                              const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                                              const unsigned int*             i_row_idx,
                                                              const unsigned int*             i_column_idx,
                                                              const void*                     i_values,
                                                              const unsigned int              i_packed_width );

LIBXSMM_API_INTERN
void libxsmm_generator_packed_spgemm_csr_bsparse_aarch64_sve_kloop( libxsmm_generated_code*            io_generated_code,
                                                                    libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                                                    const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                                    const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                                    const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                                    const libxsmm_aarch64_sve_type     i_sve_type,
                                                                    const unsigned int*                i_row_idx,
                                                                    const unsigned int*                i_column_idx,
                                                                    const void*                        i_values,
                                                                    const unsigned int                 i_n_processed,
                                                                    const unsigned int                 i_n_limit,
                                                                    const unsigned int                 i_packed_processed,
                                                                    const unsigned int                 i_packed_range,
                                                                    const unsigned int                 i_packed_blocking,
                                                                    const unsigned int                 i_packed_remainder,
                                                                    const unsigned int                 i_packed_width );

#endif /* GENERATOR_PACKE_SPGEMM_CSR_BSPARSE_AARCH64_SVE_H */

