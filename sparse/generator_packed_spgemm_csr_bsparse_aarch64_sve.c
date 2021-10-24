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

#include "generator_packed_spgemm_csr_bsparse_aarch64_sve.h"
#include "generator_aarch64_instructions.h"
#include "generator_common_aarch64.h"
#include "generator_gemm_common_aarch64.h"
#include "libxsmm_main.h"

LIBXSMM_API_INTERN
void libxsmm_generator_packed_spgemm_csr_bsparse_aarch64_sve( libxsmm_generated_code*         io_generated_code,
                                                              const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                                              const unsigned int*             i_row_idx,
                                                              const unsigned int*             i_column_idx,
                                                              const void*                     i_values,
                                                              const unsigned int              i_packed_width ) {
  unsigned int l_max_col = 0;
  unsigned int l_max_reg_block = 0;
  unsigned int l_sve_packed_full_iter = 0;
  unsigned int l_sve_packed_remainder = 0;
  unsigned int l_sve_packed_iters = 0;
  unsigned int l_sve_packed_width = 0;
  unsigned int l_packed_done = 0;
  unsigned int l_packed_count = 0; 
  unsigned int l_packed_reg_block[2] = {0, 0};
  unsigned int l_packed_reg_range[2] = {0, 0};
  unsigned int l_col_reg_block[2][2] = {{0, 0}, {0, 0}};
  unsigned int l_col_reg_range[2][2] = {{0, 0}, {0, 0}};
  unsigned int l_n;

  libxsmm_micro_kernel_config l_micro_kernel_config;
  libxsmm_loop_label_tracker l_loop_label_tracker;
  libxsmm_gp_reg_mapping l_gp_reg_mapping;
  libxsmm_aarch64_sve_type l_sve_type;

  /* define register mapping */
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

  /* define loop_label_tracker */
  libxsmm_reset_loop_label_tracker( &l_loop_label_tracker );

  /* define the micro kernel code gen properties */
  libxsmm_generator_gemm_init_micro_kernel_config_aarch64( &l_micro_kernel_config, io_generated_code->arch, i_xgemm_desc );

  /* set the type for the vector operations */
  if (LIBXSMM_GEMM_PRECISION_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype)) {
    l_sve_type = LIBXSMM_AARCH64_SVE_TYPE_D;    
  }
  else {    
    l_sve_type = LIBXSMM_AARCH64_SVE_TYPE_S;
  }

  /* set number of vector registers for blocking */
  l_max_reg_block = 30;

  /* calculate number of packed iterations */
  l_sve_packed_width = l_micro_kernel_config.vector_length;
  l_sve_packed_full_iter = i_packed_width / l_sve_packed_width;
  l_sve_packed_remainder = i_packed_width % l_sve_packed_width;
  l_sve_packed_iters = ( l_sve_packed_remainder > 0 ) ? l_sve_packed_full_iter + 1 : l_sve_packed_full_iter;

  /* determine the max index of a non-zero column */
  for ( l_n = 0; l_n < i_row_idx[i_xgemm_desc->k]; l_n++ ) {
    if (l_max_col < i_column_idx[l_n]) {
      l_max_col = i_column_idx[l_n];
    }
  }
  l_max_col++;
  
  /* calculate the distribution of the registers for packed blocking */
  libxsmm_compute_equalized_blocking( l_sve_packed_iters, l_max_reg_block, &(l_packed_reg_range[0]), &(l_packed_reg_block[0]), &(l_packed_reg_range[1]), &(l_packed_reg_block[1]) );

  /* calculate the distribution of the registers for n blocking */
  libxsmm_compute_equalized_blocking( l_max_col, l_max_reg_block/l_packed_reg_block[0], &(l_col_reg_range[0][0]), &(l_col_reg_block[0][0]), &(l_col_reg_range[0][1]), &(l_col_reg_block[0][1]) );
  
  if ( l_packed_reg_block[1] != 0 ) {
    libxsmm_compute_equalized_blocking( l_max_col, l_max_reg_block/l_packed_reg_block[1], &(l_col_reg_range[1][0]), &(l_col_reg_block[1][0]), &(l_col_reg_range[1][1]), &(l_col_reg_block[1][1]) );
  }

  /* open asm */
  libxsmm_aarch64_instruction_open_stream( io_generated_code, 0xf );

  /* create an all-true predicate register p0 */
  libxsmm_aarch64_instruction_sve_pcompute( io_generated_code, 
                                            LIBXSMM_AARCH64_INSTR_SVE_PTRUE,
                                            0,
                                            LIBXSMM_AARCH64_GP_REG_UNDEF, 
                                            LIBXSMM_AARCH64_GP_WIDTH_X, 
                                            LIBXSMM_AARCH64_GP_REG_UNDEF,
                                            LIBXSMM_AARCH64_SVE_PATTERN_ALL,
                                            LIBXSMM_AARCH64_SVE_TYPE_B);

  /* start m-loop */
  libxsmm_generator_loop_header_aarch64( io_generated_code, &l_loop_label_tracker, l_gp_reg_mapping.gp_reg_mloop, i_xgemm_desc->m );

  /* Loop over packed blocking ranges */
  while ( l_packed_done != l_sve_packed_iters ) {
    unsigned int l_packed_blocking = l_packed_reg_block[l_packed_count];
    unsigned int l_packed_remainder = 0;
    unsigned int l_n_done = 0;
    unsigned int l_n_count = 0;
    unsigned int l_n_processed = 0;

    /* adjust packed remainder */
    if (l_sve_packed_remainder != 0) {
      if (l_packed_count == 0 && l_packed_reg_block[1] > 0) {
        l_packed_remainder = 0;
      }
      else {
        l_packed_remainder = l_sve_packed_remainder;
      }
    }

    /* Loop over n blocking ranges */
    while ( l_n_done < l_max_col ) {
      unsigned int l_n_blocking = l_col_reg_block[l_packed_count][l_n_count];

      /* loop over n blocks */
      for (l_n_processed = l_n_done; l_n_processed < l_n_done + l_col_reg_range[l_packed_count][l_n_count]; l_n_processed += l_n_blocking) {
        libxsmm_generator_packed_spgemm_csr_bsparse_aarch64_sve_kloop(  io_generated_code,
                                                                        &l_loop_label_tracker,
                                                                        &l_gp_reg_mapping,
                                                                        &l_micro_kernel_config,
                                                                        i_xgemm_desc,
                                                                        l_sve_type,
                                                                        i_row_idx,
                                                                        i_column_idx,
                                                                        i_values,
                                                                        l_n_processed,
                                                                        l_n_blocking,
                                                                        l_packed_done,
                                                                        l_packed_reg_range[l_packed_count],
                                                                        l_packed_blocking,
                                                                        l_packed_remainder,
                                                                        i_packed_width );
      }
      l_n_done += l_col_reg_range[l_packed_count][l_n_count];
      l_n_count++;
    }
    l_packed_done += l_packed_reg_range[l_packed_count];
    l_packed_count++;

    if (l_packed_count == 1) {
      /* advance A pointer to the next packed range */
      libxsmm_aarch64_instruction_alu_compute_imm64(  io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                      l_gp_reg_mapping.gp_reg_a, l_gp_reg_mapping.gp_reg_help_0, l_gp_reg_mapping.gp_reg_a,
                                                      l_packed_done * l_micro_kernel_config.vector_length * l_micro_kernel_config.datatype_size_in );

      /* reset C pointer to the beginning of the row and the next packed range if the packed loop has not finished yet */
      if (l_packed_done != l_sve_packed_iters) {
        libxsmm_aarch64_instruction_alu_compute_imm64(  io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                        l_gp_reg_mapping.gp_reg_c, l_gp_reg_mapping.gp_reg_help_2, l_gp_reg_mapping.gp_reg_c,
                                                        l_micro_kernel_config.datatype_size_out*i_packed_width*i_xgemm_desc->ldc -
                                                        l_packed_done * l_micro_kernel_config.vector_length * l_micro_kernel_config.datatype_size_out );
      }
    }
  }

  /* advance A pointer to the next row */
  libxsmm_aarch64_instruction_alu_compute_imm64(  io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                  l_gp_reg_mapping.gp_reg_a, l_gp_reg_mapping.gp_reg_help_0, l_gp_reg_mapping.gp_reg_a,
                                                  l_micro_kernel_config.datatype_size_in * i_packed_width * i_xgemm_desc->lda - 
                                                  l_packed_reg_range[0] * l_micro_kernel_config.vector_length * l_micro_kernel_config.datatype_size_in );

  /* set C to the first packed block */
  if (l_packed_reg_range[0] != l_sve_packed_iters) {
    libxsmm_aarch64_instruction_alu_compute_imm64(  io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                    l_gp_reg_mapping.gp_reg_c, l_gp_reg_mapping.gp_reg_help_2, l_gp_reg_mapping.gp_reg_c,
                                                    l_packed_reg_range[0] * l_micro_kernel_config.vector_length * l_micro_kernel_config.datatype_size_out );
  }

  /* advance C pointer to the next row if some columns were skipped */
  if (l_max_col != i_xgemm_desc->n) {
    libxsmm_aarch64_instruction_alu_compute_imm64(  io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                    l_gp_reg_mapping.gp_reg_c, l_gp_reg_mapping.gp_reg_help_2, l_gp_reg_mapping.gp_reg_c,
                                                    (i_xgemm_desc->n - l_max_col) * i_packed_width * l_micro_kernel_config.datatype_size_out );
  }

  /* close m-loop */
  libxsmm_generator_loop_footer_aarch64( io_generated_code, &l_loop_label_tracker, l_gp_reg_mapping.gp_reg_mloop, 1 );

  /* close asm */
  libxsmm_aarch64_instruction_close_stream( io_generated_code, 0xf );
}

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
                                                                    const unsigned int                 i_n_blocking,
                                                                    const unsigned int                 i_packed_processed,
                                                                    const unsigned int                 i_packed_range,
                                                                    const unsigned int                 i_packed_blocking,
                                                                    const unsigned int                 i_packed_remainder,
                                                                    const unsigned int                 i_packed_width ) {
  unsigned int l_n = 0;
  unsigned int l_p = 0;
  unsigned int l_k = 0;
  unsigned int l_found_mul = 0;
  unsigned int l_n_limit = i_n_processed + i_n_blocking;
  unsigned int l_max_reg_block = i_n_blocking * i_packed_blocking;
  unsigned int l_row_elements = 0;
  unsigned int l_packed_full_iters = i_packed_range / i_packed_blocking;
  unsigned int l_loaded_elements = 0;

  LIBXSMM_UNUSED(i_values);
  LIBXSMM_ASSERT( i_packed_blocking > 0 );

  /* set predicate register p1 for remaining packed elements */
  if ( i_packed_remainder != 0 ) {
    libxsmm_aarch64_instruction_alu_set_imm64(  io_generated_code,
                                                i_gp_reg_mapping->gp_reg_help_3,
                                                i_packed_remainder );

    libxsmm_aarch64_instruction_alu_set_imm64(  io_generated_code, i_gp_reg_mapping->gp_reg_help_2, 0);

    libxsmm_aarch64_instruction_sve_pcompute( io_generated_code, 
                                              LIBXSMM_AARCH64_INSTR_SVE_WHILELT,
                                              1,
                                              i_gp_reg_mapping->gp_reg_help_2, 
                                              LIBXSMM_AARCH64_GP_WIDTH_X, 
                                              i_gp_reg_mapping->gp_reg_help_3,
                                              LIBXSMM_AARCH64_SVE_PATTERN_ALL,
                                              i_sve_type);
  }

  /* packed loop */
  if ( l_packed_full_iters> 1 ) {
    libxsmm_generator_loop_header_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_help_3, l_packed_full_iters );
  }  

  /* load C accumulator */
  for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
    for ( l_p = 0; l_p < i_packed_blocking; l_p++ ) {
      /* calculate corresponding vector register */
      unsigned int l_reg = (l_n*i_packed_blocking) + l_p;
      if (0 != (LIBXSMM_GEMM_FLAG_BETA_0 & i_xgemm_desc->flags)) { /* Beta=0 */
        /* set vector elements to 0 */
        libxsmm_aarch64_instruction_sve_compute(io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_EOR_V, l_reg, l_reg, 0, l_reg, 0, i_sve_type);
      } else {      
        /* predicated load */
        if (i_sve_type == LIBXSMM_AARCH64_SVE_TYPE_D) {
          libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_LD1D_I_OFF, 
                                                i_gp_reg_mapping->gp_reg_c, LIBXSMM_AARCH64_GP_REG_UNDEF, 0,
                                                l_reg, (l_p == i_packed_blocking-1) && (i_packed_remainder != 0) ? 1 : 0);
        } else {
          libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_LD1W_I_OFF, 
                                                i_gp_reg_mapping->gp_reg_c, LIBXSMM_AARCH64_GP_REG_UNDEF, 0,
                                                l_reg, (l_p == i_packed_blocking-1) && (i_packed_remainder != 0) ? 1 : 0);
        }   
      }
      l_loaded_elements = (l_p == i_packed_blocking-1) && (i_packed_remainder != 0) ? i_packed_remainder : i_micro_kernel_config->vector_length;
      /* advance C */
      libxsmm_aarch64_instruction_alu_compute_imm64(  io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                      i_gp_reg_mapping->gp_reg_c, i_gp_reg_mapping->gp_reg_help_2, i_gp_reg_mapping->gp_reg_c,
                                                      l_loaded_elements * i_micro_kernel_config->datatype_size_out );
    }
    /* advance C to the next column if the complete tensor length does not fit into all blocking vector registers */
    if (i_packed_blocking * i_micro_kernel_config->vector_length < i_packed_width) {
      libxsmm_aarch64_instruction_alu_compute_imm64(  io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                      i_gp_reg_mapping->gp_reg_c, i_gp_reg_mapping->gp_reg_help_2, i_gp_reg_mapping->gp_reg_c,
                                                      (i_packed_width - (i_packed_blocking - 1) * i_micro_kernel_config->vector_length - l_loaded_elements) * i_micro_kernel_config->datatype_size_out );
    }
  }
  /* reset C to the beginning of n block */
  libxsmm_aarch64_instruction_alu_compute_imm64(  io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                  i_gp_reg_mapping->gp_reg_c, i_gp_reg_mapping->gp_reg_help_2, i_gp_reg_mapping->gp_reg_c,
                                                  i_n_blocking * i_packed_width * i_micro_kernel_config->datatype_size_out );


  /* do dense packed times sparse multiplication */
  for ( l_k = 0; l_k < (unsigned int)i_xgemm_desc->k; l_k++ ) {
    l_row_elements = i_row_idx[l_k+1] - i_row_idx[l_k];
    l_found_mul = 0;
    /* check if we actually need to multiply */
    for ( l_n = 0; l_n < l_row_elements; l_n++ ) {
      if ( (i_column_idx[i_row_idx[l_k] + l_n] < (unsigned int)i_xgemm_desc->n) &&
        (i_column_idx[i_row_idx[l_k] + l_n] >= i_n_processed)        &&
        (i_column_idx[i_row_idx[l_k] + l_n] < l_n_limit) )            {
        l_found_mul = 1;
      }
    }
    /* only load A if multiplication loop is not empty */
    if (l_found_mul != 0) {
      for ( l_p = 0; l_p < i_packed_blocking; l_p++ ) {
        /* predicated load A */
        if (i_sve_type == LIBXSMM_AARCH64_SVE_TYPE_D) {
          libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_LD1D_I_OFF,
                                                i_gp_reg_mapping->gp_reg_a, LIBXSMM_AARCH64_GP_REG_UNDEF, 0,
                                                l_max_reg_block, (l_p == i_packed_blocking-1) && (i_packed_remainder != 0) ? 1 : 0);
        } else {
          libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_LD1W_I_OFF,
                                                i_gp_reg_mapping->gp_reg_a, LIBXSMM_AARCH64_GP_REG_UNDEF, 0,
                                                l_max_reg_block, (l_p == i_packed_blocking-1) && (i_packed_remainder != 0) ? 1 : 0);
        }
        l_loaded_elements = (l_p == i_packed_blocking-1) && (i_packed_remainder != 0) ? i_packed_remainder : i_micro_kernel_config->vector_length;
        /* advance A */
        libxsmm_aarch64_instruction_alu_compute_imm64(  io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD, 
                                                        i_gp_reg_mapping->gp_reg_a, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_a, 
                                                        l_loaded_elements * i_micro_kernel_config->datatype_size_in );

        /* loop over element in the row of B and multiply*/
        for ( l_n = 0; l_n < l_row_elements; l_n++ ) {
        /* check k such that we just use columns which actually need to be multiplied */
          if ( (i_column_idx[i_row_idx[l_k] + l_n] < (unsigned int)i_xgemm_desc->n) &&
            (i_column_idx[i_row_idx[l_k] + l_n] >= i_n_processed)        &&
            (i_column_idx[i_row_idx[l_k] + l_n] < l_n_limit) )            {
            
            /* broadcast B */
            if (i_sve_type == LIBXSMM_AARCH64_SVE_TYPE_D) {
              libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_LD1RD_I_OFF, 
                                                    i_gp_reg_mapping->gp_reg_b, LIBXSMM_AARCH64_GP_REG_UNDEF, 0,
                                                    l_max_reg_block + 1, 0);
            } else {
              libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_LD1RW_I_OFF, 
                                                    i_gp_reg_mapping->gp_reg_b, LIBXSMM_AARCH64_GP_REG_UNDEF, 0,
                                                    l_max_reg_block + 1, 0);
            }  
            /* FMA */
            libxsmm_aarch64_instruction_sve_compute(  io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FMLA_V, 
                                                      l_max_reg_block, l_max_reg_block + 1, 0, ((i_column_idx[i_row_idx[l_k] + l_n] - i_n_processed)*i_packed_blocking) + l_p, 
                                                      0, i_sve_type);
          }
          /* advance B to the next element in the row */
          libxsmm_aarch64_instruction_alu_compute_imm64(  io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD, 
                                                          i_gp_reg_mapping->gp_reg_b, i_gp_reg_mapping->gp_reg_help_1, i_gp_reg_mapping->gp_reg_b, 
                                                          i_micro_kernel_config->datatype_size_in );
        }
        /* reset B pointer to the beginning of the row */
        if (l_p < (i_packed_blocking - 1)) {
          libxsmm_aarch64_instruction_alu_compute_imm64(  io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB, 
                                                          i_gp_reg_mapping->gp_reg_b, i_gp_reg_mapping->gp_reg_help_1, 
                                                          i_gp_reg_mapping->gp_reg_b, l_row_elements * i_micro_kernel_config->datatype_size_in);
        } 
      }
      /* advance A to the next column if the complete tensor length does not fit into all blocking vector registers */
      if (i_packed_blocking * i_micro_kernel_config->vector_length != i_packed_width) {
        libxsmm_aarch64_instruction_alu_compute_imm64(  io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD, 
                                                        i_gp_reg_mapping->gp_reg_a, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_a, 
                                                        (i_packed_width - (i_packed_blocking - 1) * i_micro_kernel_config->vector_length - l_loaded_elements) * i_micro_kernel_config->datatype_size_in );
      }
    } else {
      /* advance A the next column if no multiplication was performed */
      libxsmm_aarch64_instruction_alu_compute_imm64(  io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD, 
                                                      i_gp_reg_mapping->gp_reg_a, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_a, 
                                                      i_packed_width * i_micro_kernel_config->datatype_size_in );
      /* advance B to the next row */
      libxsmm_aarch64_instruction_alu_compute_imm64(  io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD, 
                                                      i_gp_reg_mapping->gp_reg_b, i_gp_reg_mapping->gp_reg_help_1, 
                                                      i_gp_reg_mapping->gp_reg_b, l_row_elements * i_micro_kernel_config->datatype_size_in);
    } 
  }

  /* completly reset B pointer for next iteration  */
  libxsmm_aarch64_instruction_alu_compute_imm64(  io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB, 
                                                  i_gp_reg_mapping->gp_reg_b, i_gp_reg_mapping->gp_reg_help_1, 
                                                  i_gp_reg_mapping->gp_reg_b, i_row_idx[i_xgemm_desc->k] * i_micro_kernel_config->datatype_size_in); 

  /* store C accumulator */
  for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
    for ( l_p = 0; l_p < i_packed_blocking; l_p++ ) {
      /* calculate corresponding vector register */
      unsigned int l_reg = (l_n*i_packed_blocking) + l_p;
      /* predicated store */     
      if (i_sve_type == LIBXSMM_AARCH64_SVE_TYPE_D) {
        libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ST1D_I_OFF, 
                                              i_gp_reg_mapping->gp_reg_c, LIBXSMM_AARCH64_GP_REG_UNDEF, 0,
                                              l_reg, (l_p == i_packed_blocking-1) && (i_packed_remainder != 0) ? 1 : 0);
      } else {
        libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ST1W_I_OFF, 
                                              i_gp_reg_mapping->gp_reg_c, LIBXSMM_AARCH64_GP_REG_UNDEF, 0,
                                              l_reg, (l_p == i_packed_blocking-1) && (i_packed_remainder != 0) ? 1 : 0);
      }  
      l_loaded_elements = (l_p == i_packed_blocking-1) && (i_packed_remainder != 0) ? i_packed_remainder : i_micro_kernel_config->vector_length;
      /* advance C */
      libxsmm_aarch64_instruction_alu_compute_imm64(  io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                      i_gp_reg_mapping->gp_reg_c, i_gp_reg_mapping->gp_reg_help_2, i_gp_reg_mapping->gp_reg_c,
                                                      l_loaded_elements * i_micro_kernel_config->datatype_size_out );
    }
    /* advance C to the next column if the complete tensor length does not fit into all blocking vector registers */
    if (i_packed_blocking * i_micro_kernel_config->vector_length != i_packed_width) {
      libxsmm_aarch64_instruction_alu_compute_imm64(  io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                      i_gp_reg_mapping->gp_reg_c, i_gp_reg_mapping->gp_reg_help_2, i_gp_reg_mapping->gp_reg_c,
                                                      (i_packed_width - (i_packed_blocking - 1) * i_micro_kernel_config->vector_length - l_loaded_elements) * i_micro_kernel_config->datatype_size_out );
    }
  }

  /* packed loop */
  if ( l_packed_full_iters > 1 ) {
    /* reset A to the beginning of the row and advance A to the beginning of the next packed block */
    libxsmm_aarch64_instruction_alu_compute_imm64(  io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB, 
                                                    i_gp_reg_mapping->gp_reg_a, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_a, 
                                                    i_xgemm_desc->k * i_packed_width * i_micro_kernel_config->datatype_size_in -
                                                    ((i_packed_blocking - 1) * i_micro_kernel_config->vector_length + l_loaded_elements) * i_micro_kernel_config->datatype_size_in );
    /* reset C to the beginning of the n block and advance C to the beginning of the next packed block */
    libxsmm_aarch64_instruction_alu_compute_imm64(  io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                    i_gp_reg_mapping->gp_reg_c, i_gp_reg_mapping->gp_reg_help_2, i_gp_reg_mapping->gp_reg_c,
                                                    i_n_blocking * i_packed_width * i_micro_kernel_config->datatype_size_out -
                                                    ((i_packed_blocking - 1) * i_micro_kernel_config->vector_length + l_loaded_elements) * i_micro_kernel_config->datatype_size_in );

    /* packed loop footer */
    libxsmm_generator_loop_footer_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_help_3, 1 );

    /* reset A to the beginning of the packed range */
    libxsmm_aarch64_instruction_alu_compute_imm64(  io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB, 
                                                    i_gp_reg_mapping->gp_reg_a, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_a, 
                                                    ((i_packed_range - 1) * i_micro_kernel_config->vector_length + l_loaded_elements) * i_micro_kernel_config->datatype_size_in );
    /* advance C to the beginning of the next n block and the first packed block */
    libxsmm_aarch64_instruction_alu_compute_imm64(  io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                    i_gp_reg_mapping->gp_reg_c, i_gp_reg_mapping->gp_reg_help_2, i_gp_reg_mapping->gp_reg_c,
                                                    (i_n_blocking + 1) * i_packed_width * i_micro_kernel_config->datatype_size_out -
                                                    ((i_packed_range - 1) * i_micro_kernel_config->vector_length + l_loaded_elements) * i_micro_kernel_config->datatype_size_out );
  }
  else {
    /* reset A to the beginning of the row */
    libxsmm_aarch64_instruction_alu_compute_imm64(  io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB, 
                                                    i_gp_reg_mapping->gp_reg_a, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_a, 
                                                    i_xgemm_desc->k * i_packed_width * i_micro_kernel_config->datatype_size_in );

    /* advance C to the next column (the beginning of the next n block) if the complete tensor length does not fit into all blocking vector registers */
    if (i_packed_blocking * i_micro_kernel_config->vector_length != i_packed_width) {
      libxsmm_aarch64_instruction_alu_compute_imm64(  io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                      i_gp_reg_mapping->gp_reg_c, i_gp_reg_mapping->gp_reg_help_2, i_gp_reg_mapping->gp_reg_c,
                                                      (i_packed_width - (i_packed_blocking - 1) * i_micro_kernel_config->vector_length - l_loaded_elements) * i_micro_kernel_config->datatype_size_out );
    }
  }
}


