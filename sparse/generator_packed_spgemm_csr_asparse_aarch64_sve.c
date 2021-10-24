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

#include "generator_packed_spgemm_csr_asparse_aarch64_sve.h"
#include "generator_aarch64_instructions.h"
#include "generator_common_aarch64.h"
#include "generator_gemm_common_aarch64.h"
#include "libxsmm_main.h"

LIBXSMM_API_INTERN
void libxsmm_generator_packed_spgemm_csr_asparse_aarch64_sve( libxsmm_generated_code*         io_generated_code,
                                                              const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                                              const unsigned int*             i_row_idx,
                                                              const unsigned int*             i_column_idx,
                                                              const void*                     i_values,
                                                              const unsigned int              i_packed_width ) {
  unsigned int l_sve_packed_remainder = 0;
  unsigned int l_sve_packed_iters_full = 0;
  unsigned int l_sve_packed_width = 0;
  unsigned int l_n_max_block = 0;

  libxsmm_micro_kernel_config l_micro_kernel_config;
  libxsmm_loop_label_tracker l_loop_label_tracker;
  libxsmm_gp_reg_mapping l_gp_reg_mapping;
  libxsmm_aarch64_sve_type l_sve_type;

  /* define gp register mapping */
  libxsmm_reset_aarch64_gp_reg_mapping( &l_gp_reg_mapping );

  l_gp_reg_mapping.gp_reg_a = LIBXSMM_AARCH64_GP_REG_X0;
  l_gp_reg_mapping.gp_reg_b = LIBXSMM_AARCH64_GP_REG_X1;
  l_gp_reg_mapping.gp_reg_c = LIBXSMM_AARCH64_GP_REG_X2;
  l_gp_reg_mapping.gp_reg_a_prefetch = LIBXSMM_AARCH64_GP_REG_X3;
  l_gp_reg_mapping.gp_reg_b_prefetch = LIBXSMM_AARCH64_GP_REG_X4;
  l_gp_reg_mapping.gp_reg_mloop = LIBXSMM_AARCH64_GP_REG_X6;
  l_gp_reg_mapping.gp_reg_nloop = LIBXSMM_AARCH64_GP_REG_X7;
  l_gp_reg_mapping.gp_reg_kloop = LIBXSMM_AARCH64_GP_REG_X8;
  l_gp_reg_mapping.gp_reg_help_0 = LIBXSMM_AARCH64_GP_REG_X9;
  l_gp_reg_mapping.gp_reg_help_1 = LIBXSMM_AARCH64_GP_REG_X10;
  l_gp_reg_mapping.gp_reg_help_2 = LIBXSMM_AARCH64_GP_REG_X11;
  l_gp_reg_mapping.gp_reg_help_3 = LIBXSMM_AARCH64_GP_REG_X12;
  l_gp_reg_mapping.gp_reg_help_4 = LIBXSMM_AARCH64_GP_REG_UNDEF;
  l_gp_reg_mapping.gp_reg_help_5 = LIBXSMM_AARCH64_GP_REG_UNDEF;

  /* define the micro kernel code gen properties */
  libxsmm_generator_gemm_init_micro_kernel_config_aarch64( &l_micro_kernel_config, io_generated_code->arch, i_xgemm_desc );

  /* set the type for SVE operations */
  if (LIBXSMM_GEMM_PRECISION_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype)) {
    l_sve_type = LIBXSMM_AARCH64_SVE_TYPE_D;    
  }
  else {    
    l_sve_type = LIBXSMM_AARCH64_SVE_TYPE_S;
  }

  /* select packed width */
  l_sve_packed_width = l_micro_kernel_config.vector_length;

  /* calculate the packing count */
  l_sve_packed_remainder = i_packed_width % l_sve_packed_width;
  l_sve_packed_iters_full = i_packed_width / l_sve_packed_width;    

  /* select N blocking width as we have 32 vector registers */
  l_n_max_block = 30;

  /* define loop_label_tracker */
  libxsmm_reset_loop_label_tracker( &l_loop_label_tracker );

  /* open asm */
  libxsmm_aarch64_instruction_open_stream( io_generated_code, 0xf );

  /* create an all-true predicate register */
  libxsmm_aarch64_instruction_sve_pcompute( io_generated_code, 
                                            LIBXSMM_AARCH64_INSTR_SVE_PTRUE,
                                            0,
                                            LIBXSMM_AARCH64_GP_REG_UNDEF, 
                                            LIBXSMM_AARCH64_GP_WIDTH_X, 
                                            LIBXSMM_AARCH64_GP_REG_UNDEF,
                                            LIBXSMM_AARCH64_SVE_PATTERN_ALL,
                                            LIBXSMM_AARCH64_SVE_TYPE_B);

  /* loop over blocks of packing */
  if ( (l_sve_packed_iters_full > 1) || (l_sve_packed_remainder > 0 && l_sve_packed_iters_full > 0 ) ) {
    libxsmm_generator_loop_header_aarch64( io_generated_code, &l_loop_label_tracker, l_gp_reg_mapping.gp_reg_help_0, l_sve_packed_iters_full );

    /* save pointers for outer loop */
    libxsmm_aarch64_instruction_alu_compute_imm12(  io_generated_code, LIBXSMM_AARCH64_INSTR_GP_SUB_I,
                                                    LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_XSP,
                                                    32, 0 );
    libxsmm_aarch64_instruction_alu_pair_move(  io_generated_code, LIBXSMM_AARCH64_INSTR_GP_STP_I_OFF, LIBXSMM_AARCH64_GP_REG_XSP, 16,
                                                l_gp_reg_mapping.gp_reg_a, l_gp_reg_mapping.gp_reg_b );
    libxsmm_aarch64_instruction_alu_pair_move(  io_generated_code, LIBXSMM_AARCH64_INSTR_GP_STP_I_OFF, LIBXSMM_AARCH64_GP_REG_XSP, 0,
                                                l_gp_reg_mapping.gp_reg_c, l_gp_reg_mapping.gp_reg_b_prefetch );
  }

  /* call N loop */
  if ( l_sve_packed_iters_full > 0 ) {
    libxsmm_generator_packed_spgemm_csr_asparse_aarch64_sve_n_loop( io_generated_code,
                                                                    i_xgemm_desc,
                                                                    &l_loop_label_tracker,
                                                                    &l_micro_kernel_config,
                                                                    &l_gp_reg_mapping,
                                                                    l_sve_type,
                                                                    i_row_idx,
                                                                    i_column_idx,
                                                                    i_values,
                                                                    l_n_max_block,
                                                                    i_packed_width,
                                                                    0 );
  }

  /* close packed loop */
  if ( (l_sve_packed_iters_full > 1) || (l_sve_packed_remainder > 0 && l_sve_packed_iters_full > 0 ) ) {
    /* restore pointers from stack */
    libxsmm_aarch64_instruction_alu_pair_move(  io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDP_I_OFF, LIBXSMM_AARCH64_GP_REG_XSP, 16,
                                                l_gp_reg_mapping.gp_reg_a, l_gp_reg_mapping.gp_reg_b );
    libxsmm_aarch64_instruction_alu_pair_move(  io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDP_I_OFF, LIBXSMM_AARCH64_GP_REG_XSP, 0,
                                                l_gp_reg_mapping.gp_reg_c, l_gp_reg_mapping.gp_reg_b_prefetch );
    libxsmm_aarch64_instruction_alu_compute_imm12(  io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I,
                                                    LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_XSP,
                                                    32, 0 );

    /* advance B and C pointers to the next packed block */
    libxsmm_aarch64_instruction_alu_compute_imm64(  io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                    l_gp_reg_mapping.gp_reg_c, l_gp_reg_mapping.gp_reg_help_2, l_gp_reg_mapping.gp_reg_c,
                                                    l_sve_packed_width*l_micro_kernel_config.datatype_size_out );
    libxsmm_aarch64_instruction_alu_compute_imm64(  io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                    l_gp_reg_mapping.gp_reg_b, l_gp_reg_mapping.gp_reg_help_1, l_gp_reg_mapping.gp_reg_b,
                                                    l_sve_packed_width*l_micro_kernel_config.datatype_size_in );

    libxsmm_generator_loop_footer_aarch64( io_generated_code, &l_loop_label_tracker, l_gp_reg_mapping.gp_reg_help_0, 1 );
  }

  /* handle remaining packed elements */
  if ( l_sve_packed_remainder > 0 ) {

    /* create a fitting predicate register */
    libxsmm_aarch64_instruction_alu_set_imm64(  io_generated_code,
                                                l_gp_reg_mapping.gp_reg_help_3,
                                                l_sve_packed_remainder );

    libxsmm_aarch64_instruction_alu_set_imm64(  io_generated_code, l_gp_reg_mapping.gp_reg_help_2, 0);

    libxsmm_aarch64_instruction_sve_pcompute( io_generated_code, 
                                              LIBXSMM_AARCH64_INSTR_SVE_WHILELT,
                                              1,
                                              l_gp_reg_mapping.gp_reg_help_2, 
                                              LIBXSMM_AARCH64_GP_WIDTH_X, 
                                              l_gp_reg_mapping.gp_reg_help_3,
                                              LIBXSMM_AARCH64_SVE_PATTERN_ALL,
                                              l_sve_type);

    /* call N-loop */
    libxsmm_generator_packed_spgemm_csr_asparse_aarch64_sve_n_loop( io_generated_code,
                                                                    i_xgemm_desc, 
                                                                    &l_loop_label_tracker,
                                                                    &l_micro_kernel_config, 
                                                                    &l_gp_reg_mapping, 
                                                                    l_sve_type, 
                                                                    i_row_idx, 
                                                                    i_column_idx, 
                                                                    i_values, 
                                                                    l_n_max_block, 
                                                                    l_sve_packed_width, 
                                                                    1);
  }

  /* close asm */
  libxsmm_aarch64_instruction_close_stream( io_generated_code, 0xf );
}

LIBXSMM_API_INTERN
void libxsmm_generator_packed_spgemm_csr_asparse_aarch64_sve_n_loop(  libxsmm_generated_code*            io_generated_code,
                                                                      const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                                      libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                                                      const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                                      const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                                      const libxsmm_aarch64_sve_type     i_sve_type,
                                                                      const unsigned int*                i_row_idx,
                                                                      const unsigned int*                i_column_idx,
                                                                      const void*                        i_values,
                                                                      const unsigned int                 i_n_max_block,
                                                                      const unsigned int                 i_packed_width,
                                                                      const unsigned int                 i_sve_pred_reg  ) {
  unsigned int l_gen_m_trips = 0;
  unsigned int l_a_is_dense = 0;
  unsigned int l_n_chunks = 0;
  unsigned int l_n_chunksize = 0;
  unsigned int l_n_remain = 0;
  unsigned int l_n_loop = 0;

  /* test if we should generate a dense version */
  if ( i_row_idx[i_xgemm_desc->m] == (unsigned int)(i_xgemm_desc->m*i_xgemm_desc->k) ) {
    l_gen_m_trips = 1;
    l_a_is_dense = 1;
  } else {
    l_gen_m_trips = i_xgemm_desc->m;
    l_a_is_dense = 0;
  }

  /* calculate the chunk size of current columns to work on */
  l_n_chunks = ( (i_xgemm_desc->n % i_n_max_block) == 0 ) ? (i_xgemm_desc->n / i_n_max_block) : (i_xgemm_desc->n / i_n_max_block) + 1;
  l_n_chunksize = ( (i_xgemm_desc->n % l_n_chunks) == 0 ) ? (i_xgemm_desc->n / l_n_chunks) : (i_xgemm_desc->n / l_n_chunks) + 1;
  l_n_remain = ( ((i_xgemm_desc->n % l_n_chunksize) == 0) || ((unsigned int)i_xgemm_desc->n <= i_n_max_block) ) ? 0 : 1;
  l_n_loop = ( l_n_remain == 0 ) ? (l_n_chunks * l_n_chunksize) : ((l_n_chunks-1) * l_n_chunksize);

  /* loop over blocks of n */
  libxsmm_generator_loop_header_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_nloop, l_n_loop );

  /* do matix multiplication for a block of N columns */
  libxsmm_generator_packed_spgemm_csr_asparse_aarch64_sve_m_loop( io_generated_code, i_xgemm_desc, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping,
                                                                  i_sve_type, i_row_idx, i_column_idx, i_values,
                                                                  l_gen_m_trips, l_a_is_dense, l_n_chunksize, i_packed_width, i_sve_pred_reg );

  /* advance B pointer to the next n-chunk */
  libxsmm_aarch64_instruction_alu_compute_imm64(  io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                  i_gp_reg_mapping->gp_reg_b, i_gp_reg_mapping->gp_reg_help_1, i_gp_reg_mapping->gp_reg_b,
                                                  i_micro_kernel_config->datatype_size_in*i_packed_width*l_n_chunksize );

  /* advance C pointer to the next n-chunk */
  libxsmm_aarch64_instruction_alu_compute_imm64(  io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                  i_gp_reg_mapping->gp_reg_c, i_gp_reg_mapping->gp_reg_help_2, i_gp_reg_mapping->gp_reg_c,
                                                  (unsigned long long)((unsigned long long)(i_micro_kernel_config->datatype_size_out*i_packed_width*i_xgemm_desc->ldc*i_xgemm_desc->m)
                                                  -(i_micro_kernel_config->datatype_size_out*i_packed_width*l_n_chunksize) ) );

  /* N loop jump back */
  libxsmm_generator_loop_footer_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_nloop, l_n_chunksize );

  /* handle remainder of N loop */
  if ( l_n_remain != 0 ) {
    libxsmm_generator_packed_spgemm_csr_asparse_aarch64_sve_m_loop( io_generated_code, i_xgemm_desc, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping,
                                                                    i_sve_type, i_row_idx, i_column_idx, i_values,
                                                                    l_gen_m_trips, l_a_is_dense, i_xgemm_desc->n - (l_n_chunksize * (l_n_chunks - 1)), i_packed_width, i_sve_pred_reg );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_packed_spgemm_csr_asparse_aarch64_sve_m_loop(  libxsmm_generated_code*            io_generated_code,
                                                                      const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                                      libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                                                      const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                                      const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                                      const libxsmm_aarch64_sve_type     i_sve_type,
                                                                      const unsigned int*                i_row_idx,
                                                                      const unsigned int*                i_column_idx,
                                                                      const void*                        i_values,
                                                                      const unsigned int                 i_gen_m_trips,
                                                                      const unsigned int                 i_a_is_dense,
                                                                      const unsigned int                 i_num_c_cols,
                                                                      const unsigned int                 i_packed_width,
                                                                      const unsigned int                 i_sve_pred_reg ) {
  unsigned int l_m;
  unsigned int l_n;
  unsigned int l_z;
  unsigned int l_row_elements;
  unsigned int l_b_offset;

  LIBXSMM_UNUSED(i_values);

  /* do sparse times dense packed multiplication */
  for ( l_m = 0; l_m < i_gen_m_trips; l_m++ ) {
    /* handle b offset */
    l_b_offset = 0;

    /* generate M loop */
    if (i_a_is_dense != 0 ) {
      libxsmm_generator_loop_header_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_mloop, i_xgemm_desc->m );
    }

    l_row_elements = i_row_idx[l_m+1] - i_row_idx[l_m];
    if (l_row_elements > 0) {
      /* load C accumulator */
      for ( l_n = 0; l_n < i_num_c_cols; l_n++ ) {
        if (0 != (LIBXSMM_GEMM_FLAG_BETA_0 & i_xgemm_desc->flags)) { /* Beta=0 */
          /* set all vector elements to 0 */
          libxsmm_aarch64_instruction_sve_compute(  io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_EOR_V,
                                                    l_n, l_n, 0, l_n, i_sve_pred_reg, i_sve_type);
        } else {
          /* predicated load */
          if (i_sve_type == LIBXSMM_AARCH64_SVE_TYPE_D) {
            libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_LD1D_I_OFF, 
                                                  i_gp_reg_mapping->gp_reg_c, LIBXSMM_AARCH64_GP_REG_UNDEF, 
                                                  0, l_n, i_sve_pred_reg);
          } else {
            libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_LD1W_SR, 
                                                  i_gp_reg_mapping->gp_reg_c, LIBXSMM_AARCH64_GP_REG_UNDEF, 
                                                  0, l_n, i_sve_pred_reg);
          }
        }
        /* advance C pointer to the next column */
        libxsmm_aarch64_instruction_alu_compute_imm64(  io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD, 
                                                        i_gp_reg_mapping->gp_reg_c, i_gp_reg_mapping->gp_reg_help_3, 
                                                        i_gp_reg_mapping->gp_reg_c, i_micro_kernel_config->datatype_size_out * i_packed_width);
      }

      /* reset C pointer to the beginning of the chunk */
      libxsmm_aarch64_instruction_alu_compute_imm64(  io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB, 
                                                      i_gp_reg_mapping->gp_reg_c, i_gp_reg_mapping->gp_reg_help_3, i_gp_reg_mapping->gp_reg_c, 
                                                      i_micro_kernel_config->datatype_size_out * i_packed_width * i_num_c_cols);

      /* loop over the non-zeros in A row m */
      for ( l_z = 0; l_z < l_row_elements; l_z++ ) {
        /* broadcast values of A */
        if (i_sve_type == LIBXSMM_AARCH64_SVE_TYPE_D) {
          libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_LD1RD_I_OFF, 
                                                i_gp_reg_mapping->gp_reg_a, LIBXSMM_AARCH64_GP_REG_UNDEF, 0,
                                                i_num_c_cols, i_sve_pred_reg);
        } else {
          libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_LD1RW_I_OFF, 
                                                i_gp_reg_mapping->gp_reg_a, LIBXSMM_AARCH64_GP_REG_UNDEF, 0,
                                                i_num_c_cols, i_sve_pred_reg);
        } 
        /* advance A pointer to the next element */
        libxsmm_aarch64_instruction_alu_compute_imm64(  io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD, i_gp_reg_mapping->gp_reg_a, 
                                                        i_gp_reg_mapping->gp_reg_help_2, i_gp_reg_mapping->gp_reg_a, i_micro_kernel_config->datatype_size_in );

        /* multiply with B */
        for ( l_n = 0; l_n < i_num_c_cols; l_n++ ) {
          /* set B to the correct elements */
          l_b_offset = ((i_column_idx[i_row_idx[l_m] + l_z]*i_micro_kernel_config->datatype_size_in*i_packed_width*i_xgemm_desc->ldb)
                        +(l_n*i_packed_width*i_micro_kernel_config->datatype_size_in));

          libxsmm_aarch64_instruction_alu_set_imm64(  io_generated_code, i_gp_reg_mapping->gp_reg_help_1,
                                                      (unsigned long long)l_b_offset );

          libxsmm_aarch64_instruction_alu_compute_imm64(  io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD, 
                                                          i_gp_reg_mapping->gp_reg_b, i_gp_reg_mapping->gp_reg_help_3, 
                                                          i_gp_reg_mapping->gp_reg_b, l_b_offset);

          /* load sequence of B */
          if ( i_sve_type == LIBXSMM_AARCH64_SVE_TYPE_D ) {
            libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_LD1D_I_OFF, 
                                                  i_gp_reg_mapping->gp_reg_b, LIBXSMM_AARCH64_GP_REG_UNDEF, 0,
                                                  i_num_c_cols + 1, i_sve_pred_reg);
          }
          else {
            libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_LD1W_I_OFF, 
                                                  i_gp_reg_mapping->gp_reg_b, LIBXSMM_AARCH64_GP_REG_UNDEF, 0,
                                                  i_num_c_cols + 1, i_sve_pred_reg);
          }

          /* reset B pointer */
          libxsmm_aarch64_instruction_alu_compute_imm64(  io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB, 
                                                          i_gp_reg_mapping->gp_reg_b, i_gp_reg_mapping->gp_reg_help_3, 
                                                          i_gp_reg_mapping->gp_reg_b, l_b_offset);

          /* FMA */
          libxsmm_aarch64_instruction_sve_compute(  io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FMLA_V, 
                                                    i_num_c_cols, i_num_c_cols + 1, 0, l_n, i_sve_pred_reg, i_sve_type);
        }
      }
      /* store C accumulator */
      for ( l_n = 0; l_n < i_num_c_cols; l_n++ ) {
        /* offset for C */
        libxsmm_aarch64_instruction_alu_set_imm64(  io_generated_code, i_gp_reg_mapping->gp_reg_help_3, 
                                                    i_micro_kernel_config->datatype_size_out * i_packed_width * l_n);
        /* predicated store */
        if (i_sve_type == LIBXSMM_AARCH64_SVE_TYPE_D) {
          libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ST1D_I_OFF, 
                                                i_gp_reg_mapping->gp_reg_c, LIBXSMM_AARCH64_GP_REG_UNDEF, 
                                                0, l_n, i_sve_pred_reg);
        } else {
          libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ST1W_I_OFF, 
                                                i_gp_reg_mapping->gp_reg_c, LIBXSMM_AARCH64_GP_REG_UNDEF, 
                                                0, l_n, i_sve_pred_reg);
        }

        /* advance C pointer to the next column */
        libxsmm_aarch64_instruction_alu_compute_imm64(  io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD, 
                                                        i_gp_reg_mapping->gp_reg_c, i_gp_reg_mapping->gp_reg_help_3, 
                                                        i_gp_reg_mapping->gp_reg_c, i_micro_kernel_config->datatype_size_out * i_packed_width);
      }
      /* advance C pointer to the next row */
      libxsmm_aarch64_instruction_alu_compute_imm64(  io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD, 
                                                      i_gp_reg_mapping->gp_reg_c, i_gp_reg_mapping->gp_reg_help_3, i_gp_reg_mapping->gp_reg_c,
                                                      i_micro_kernel_config->datatype_size_out * i_packed_width * (i_xgemm_desc->ldc - i_num_c_cols));
    } else {
      /* advance C pointer to the next row */
      libxsmm_aarch64_instruction_alu_compute_imm64(  io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                      i_gp_reg_mapping->gp_reg_c, i_gp_reg_mapping->gp_reg_help_2, i_gp_reg_mapping->gp_reg_c,
                                                      i_micro_kernel_config->datatype_size_out*i_packed_width*i_xgemm_desc->ldc );
    }

    /* generate M loop */
    if (i_a_is_dense != 0 ) {
      /* M loop jump back */
      libxsmm_generator_loop_footer_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_mloop, 1 );
    }
  }

  /* reset A pointer */
  if (i_a_is_dense != 0 ) {
    libxsmm_aarch64_instruction_alu_compute_imm64(  io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                    i_gp_reg_mapping->gp_reg_a, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_a,
                                                    i_micro_kernel_config->datatype_size_in*i_xgemm_desc->k*i_xgemm_desc->m );
  } else {
    libxsmm_aarch64_instruction_alu_compute_imm64(  io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                    i_gp_reg_mapping->gp_reg_a, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_a,
                                                    i_micro_kernel_config->datatype_size_in*i_row_idx[i_gen_m_trips] );
  }
}
