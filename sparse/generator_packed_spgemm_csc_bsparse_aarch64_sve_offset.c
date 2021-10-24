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

#include "generator_packed_spgemm_csc_bsparse_aarch64_sve.h"
#include "generator_aarch64_instructions.h"
#include "generator_common_aarch64.h"
#include "generator_gemm_common_aarch64.h"
#include "libxsmm_main.h"

LIBXSMM_API_INTERN
void libxsmm_generator_packed_spgemm_csc_bsparse_aarch64_sve(   libxsmm_generated_code*     io_generated_code,
                                const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                const unsigned int*       i_row_idx,
                                const unsigned int*       i_column_idx,
                                const void*           i_values,
                                const unsigned int        i_packed_width ) {
  unsigned int l_n = 0;
  unsigned int l_max_cols = 0;
  unsigned int l_max_reg_block = 0;
  unsigned int l_sve_packed_remainder = 0;
  unsigned int l_sve_packed_iters = 0;
  unsigned int l_sve_packed_iters_full = 0;
  unsigned int l_sve_packed_width = 0;
  unsigned int l_packed_done = 0;
  unsigned int l_packed_count = 0;
  unsigned int l_packed_reg_block[2] = {0,0};
  unsigned int l_packed_reg_range[2] = {0,0};
  unsigned int l_col_reg_block[2][2] = { {0,0}, {0,0} };
  unsigned int l_col_reg_range[2][2] = { {0,0}, {0,0} };

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

  /* define loop_label_tracker */
  libxsmm_reset_loop_label_tracker( &l_loop_label_tracker );

  /* define the micro kernel code gen properties */
  libxsmm_generator_gemm_init_micro_kernel_config_aarch64( &l_micro_kernel_config, io_generated_code->arch, i_xgemm_desc );

  /* select sve type for sve operations */
  if ( LIBXSMM_GEMM_PRECISION_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype )  ) {
    l_sve_type = LIBXSMM_AARCH64_SVE_TYPE_D;
  } else {
    l_sve_type = LIBXSMM_AARCH64_SVE_TYPE_S;    
  }

  /* define the max number of registers for blocking */
  l_max_reg_block = 30;
  
  /* calculate packed blocks */
  l_sve_packed_width = l_micro_kernel_config.vector_length;
  l_sve_packed_remainder = i_packed_width % l_sve_packed_width;
  l_sve_packed_iters_full = i_packed_width / l_sve_packed_width;
  l_sve_packed_iters = ( l_sve_packed_remainder > 0 ) ? l_sve_packed_iters_full+1 : l_sve_packed_iters_full;

  /* determine the max index of a non-zero column */
  l_max_cols = i_xgemm_desc->n;
  for ( l_n = 0; l_n < i_xgemm_desc->n; l_n++ ) {
    if ( i_column_idx[l_n] == i_column_idx[i_xgemm_desc->n] ) {
      l_max_cols = l_n+1;
    }
  }

  /* calculate the distribution of the registers for packed blocking */
  libxsmm_compute_equalized_blocking( l_sve_packed_iters, l_max_reg_block, &(l_packed_reg_range[0]), &(l_packed_reg_block[0]), &(l_packed_reg_range[1]), &(l_packed_reg_block[1]) );

  /* calculate the distribution of the remaining registers for n blocking */
  libxsmm_compute_equalized_blocking( l_max_cols, l_max_reg_block/l_packed_reg_block[0], &(l_col_reg_range[0][0]), &(l_col_reg_block[0][0]), &(l_col_reg_range[0][1]), &(l_col_reg_block[0][1]) );
  if ( l_packed_reg_block[1] != 0 ) {
    libxsmm_compute_equalized_blocking( l_max_cols, l_max_reg_block/l_packed_reg_block[1], &(l_col_reg_range[1][0]), &(l_col_reg_block[1][0]), &(l_col_reg_range[1][1]), &(l_col_reg_block[1][1]) );
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

  /* m loop */
  libxsmm_generator_loop_header_aarch64( io_generated_code, &l_loop_label_tracker, l_gp_reg_mapping.gp_reg_mloop, i_xgemm_desc->m );

  /* loop over packed blocking ranges */
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

    /* loop over n blocking ranges */
    while ( l_n_done < l_max_cols ) {
      unsigned int l_n_blocking = l_col_reg_block[l_packed_count][l_n_count];

      /* loop over n blocks */
      for ( l_n_processed = l_n_done; l_n_processed < l_n_done + l_col_reg_range[l_packed_count][l_n_count]; l_n_processed += l_n_blocking ) {
        libxsmm_generator_packed_spgemm_csc_bsparse_aarch64_sve_kloop(  io_generated_code,
                                                                        &l_loop_label_tracker,
                                                                        &l_gp_reg_mapping,
                                                                        &l_micro_kernel_config,
                                                                        i_xgemm_desc,
                                                                        l_sve_type,
                                                                        i_row_idx,
                                                                        i_column_idx,
                                                                        i_values,
                                                                        l_n_processed,
                                                                        l_n_processed + l_n_blocking,
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
  }

  /* advance C pointer to the next row */
  libxsmm_aarch64_instruction_alu_compute_imm64(  io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                  l_gp_reg_mapping.gp_reg_c, l_gp_reg_mapping.gp_reg_help_2, l_gp_reg_mapping.gp_reg_c,
                                                  l_micro_kernel_config.datatype_size_out*i_packed_width*i_xgemm_desc->ldc );

  /* advance A pointer to the next row */
  libxsmm_aarch64_instruction_alu_compute_imm64(  io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                  l_gp_reg_mapping.gp_reg_a, l_gp_reg_mapping.gp_reg_help_0, l_gp_reg_mapping.gp_reg_a,
                                                  l_micro_kernel_config.datatype_size_in*i_packed_width*i_xgemm_desc->lda );

  /* close m loop */
  libxsmm_generator_loop_footer_aarch64( io_generated_code, &l_loop_label_tracker, l_gp_reg_mapping.gp_reg_mloop, 1 );

  /* close asm */
  libxsmm_aarch64_instruction_close_stream( io_generated_code, 0xf );
}

LIBXSMM_API_INTERN
void libxsmm_generator_packed_spgemm_csc_bsparse_aarch64_sve_kloop( libxsmm_generated_code*            io_generated_code,
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
                                                                    const unsigned int                 i_packed_width ) {
  unsigned int l_n = 0;
  unsigned int l_p = 0;
  unsigned int l_k = 0;
  unsigned int l_found_mul = 0;
  unsigned int l_max_reg_block = (i_n_limit - i_n_processed) * i_packed_blocking;
  unsigned int l_n_blocking = i_n_limit - i_n_processed;
  unsigned int l_packed_full_iters = i_packed_range / i_packed_blocking;

  LIBXSMM_UNUSED(i_values);
  LIBXSMM_ASSERT( i_packed_blocking > 0 );

  /* set predicate register p1 for remaining packed elements */
  if ( i_packed_remainder != 0 ) {
    libxsmm_aarch64_instruction_alu_set_imm64(  io_generated_code,
                                                i_gp_reg_mapping->gp_reg_help_3,
                                                i_packed_remainder );

    libxsmm_aarch64_instruction_alu_set_imm64(  io_generated_code, 
                                                i_gp_reg_mapping->gp_reg_help_2, 
                                                0);

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
  if ( l_packed_full_iters > 1 ) {
    libxsmm_generator_loop_header_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_help_3, l_packed_full_iters );
  }

  /* load C accumulator */
  for ( l_n = 0; l_n < l_n_blocking; l_n++ ) {
    for ( l_p = 0; l_p < i_packed_blocking; l_p++ ) {
      /* calculate corresponding vector register */
      unsigned int l_reg = (l_n*i_packed_blocking) + l_p;
      if (0 != (LIBXSMM_GEMM_FLAG_BETA_0 & i_xgemm_desc->flags)) { /* Beta=0 */
        /* set vector elements to 0 */
        libxsmm_aarch64_instruction_sve_compute(io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_EOR_V, l_reg, l_reg, 0, l_reg, 0, i_sve_type);
      } else {
        /* set register to the necessary offset */
        libxsmm_aarch64_instruction_alu_set_imm64(  io_generated_code, i_gp_reg_mapping->gp_reg_help_2, 
                                                    ( (i_n_processed + l_n)*i_packed_width ) +
                                                    ( (i_packed_processed + l_p)*i_micro_kernel_config->vector_length ));

        /* predicated load */
        if (i_sve_type == LIBXSMM_AARCH64_SVE_TYPE_D) {
          libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_LD1D_SR, 
                                                i_gp_reg_mapping->gp_reg_c, i_gp_reg_mapping->gp_reg_help_2, 0,
                                                l_reg, (l_p == i_packed_blocking-1) && (i_packed_remainder != 0) ? 1 : 0);
        } else {
          libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_LD1W_SR, 
                                                i_gp_reg_mapping->gp_reg_c, i_gp_reg_mapping->gp_reg_help_2, 0,
                                                l_reg, (l_p == i_packed_blocking-1) && (i_packed_remainder != 0) ? 1 : 0);
        }  
      }
    }
  }

  /* do dense packed times sparse multiplication */
  for ( l_k = 0; l_k < (unsigned int)i_xgemm_desc->k; l_k++ ) {
    unsigned int l_col_k = 0;
    /* number of elements from the last block to the current */
    int l_nnz_idx[28] = {0};

    /* reset helpers */
    for ( l_n = 0; l_n < l_n_blocking; l_n++ ) {
      l_nnz_idx[l_n] = -1;
    }
    l_found_mul = 0;

    /* loop over the columns of B/C */
    for ( l_n = 0; l_n < l_n_blocking; l_n++ ) {
      unsigned int l_col_elements = i_column_idx[i_n_processed+l_n+1] - i_column_idx[i_n_processed+l_n];
      unsigned int l_cur_column = i_column_idx[i_n_processed+l_n];
      /* search for entries matching that k */
      for ( l_col_k = 0; l_col_k < l_col_elements; l_col_k++ ) {
        if ( l_k == i_row_idx[l_cur_column + l_col_k] ) {
          l_nnz_idx[l_n] = l_cur_column + l_col_k;
          l_col_k = l_col_elements;
        }
      }
      /* let's check if we have an entry in the column that matches the k from A */
      if ( (l_nnz_idx[l_n] != -1) ) {
        l_found_mul = 1;
      }
    }

    if ( l_found_mul != 0 ) {
      /* loop over packed block */
      for ( l_p = 0; l_p < i_packed_blocking; l_p++ ) {
        /* set register to the necessary offset */
        libxsmm_aarch64_instruction_alu_set_imm64(  io_generated_code, i_gp_reg_mapping->gp_reg_help_0, 
                                                    l_k*i_packed_width +
                                                    (i_packed_processed + l_p)*i_micro_kernel_config->vector_length );

        /* predicated load A */
        if (i_sve_type == LIBXSMM_AARCH64_SVE_TYPE_D) {
          libxsmm_aarch64_instruction_sve_move(   io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_LD1D_SR,
                                                  i_gp_reg_mapping->gp_reg_a, i_gp_reg_mapping->gp_reg_help_0, 0,
                                                  l_max_reg_block, (l_p == i_packed_blocking-1) && (i_packed_remainder != 0) ? 1 : 0);
        } else {
          libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_LD1W_SR,
                                                i_gp_reg_mapping->gp_reg_a, i_gp_reg_mapping->gp_reg_help_0, 0,
                                                l_max_reg_block, (l_p == i_packed_blocking-1) && (i_packed_remainder != 0) ? 1 : 0);
        }

        /* loop over the columns of B/C */
        for ( l_n = 0; l_n < l_n_blocking; l_n++ ) {
          if ( l_nnz_idx[l_n] != -1 ) {
            /* advance B pointer to the next element */
            libxsmm_aarch64_instruction_alu_compute_imm64(  io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD, 
                                                            i_gp_reg_mapping->gp_reg_b, i_gp_reg_mapping->gp_reg_help_1, 
                                                            i_gp_reg_mapping->gp_reg_b, l_nnz_idx[l_n] * i_micro_kernel_config->datatype_size_in);
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
                                                      l_max_reg_block, l_max_reg_block + 1, 0, (l_n*i_packed_blocking) + l_p, 
                                                      0, i_sve_type);
            /* reset B pointer */
            libxsmm_aarch64_instruction_alu_compute_imm64(  io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB, 
                                                            i_gp_reg_mapping->gp_reg_b, i_gp_reg_mapping->gp_reg_help_1, 
                                                            i_gp_reg_mapping->gp_reg_b, l_nnz_idx[l_n] * i_micro_kernel_config->datatype_size_in);
          }
        }
      }
    }
  }

  /* store C accumulator */
  for ( l_n = 0; l_n < l_n_blocking; l_n++ ) {
    for ( l_p = 0; l_p < i_packed_blocking; l_p++ ) {
      /* calculate corresponding vector register */
      unsigned int l_reg = (l_n*i_packed_blocking) + l_p;

      /* set register to the necessary offset */
      libxsmm_aarch64_instruction_alu_set_imm64(  io_generated_code, i_gp_reg_mapping->gp_reg_help_2, 
                                                  ( (i_n_processed + l_n)*i_packed_width ) +
                                                  ( (i_packed_processed + l_p)*i_micro_kernel_config->vector_length ));

      /* predicated store */     
      if (i_sve_type == LIBXSMM_AARCH64_SVE_TYPE_D) {
        libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ST1D_SR, 
                                              i_gp_reg_mapping->gp_reg_c, i_gp_reg_mapping->gp_reg_help_2, 0,
                                              l_reg, (l_p == i_packed_blocking-1) && (i_packed_remainder != 0) ? 1 : 0);
      } else {
        libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ST1W_SR, 
                                              i_gp_reg_mapping->gp_reg_c, i_gp_reg_mapping->gp_reg_help_2, 0,
                                              l_reg, (l_p == i_packed_blocking-1) && (i_packed_remainder != 0) ? 1 : 0);
      } 
    }
  }

  /* packed loop */
  if ( l_packed_full_iters > 1 ) {
    /* advance A and C pointer to the next packed block */
    libxsmm_aarch64_instruction_alu_compute_imm64(  io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                    i_gp_reg_mapping->gp_reg_a, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_a,
                                                    i_packed_blocking*i_micro_kernel_config->vector_length*i_micro_kernel_config->datatype_size_in );
    libxsmm_aarch64_instruction_alu_compute_imm64(  io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                    i_gp_reg_mapping->gp_reg_c, i_gp_reg_mapping->gp_reg_help_2, i_gp_reg_mapping->gp_reg_c,
                                                    i_packed_blocking*i_micro_kernel_config->vector_length*i_micro_kernel_config->datatype_size_out );

    /* packed loop footer */
    libxsmm_generator_loop_footer_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_help_3, 1 );

    /* reset A and C pointer to the beginning of the packed range */
    libxsmm_aarch64_instruction_alu_compute_imm64(  io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                    i_gp_reg_mapping->gp_reg_a, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_a,
                                                    i_packed_range*i_micro_kernel_config->vector_length*i_micro_kernel_config->datatype_size_in );
    libxsmm_aarch64_instruction_alu_compute_imm64(  io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                    i_gp_reg_mapping->gp_reg_c, i_gp_reg_mapping->gp_reg_help_2, i_gp_reg_mapping->gp_reg_c,
                                                    i_packed_range*i_micro_kernel_config->vector_length*i_micro_kernel_config->datatype_size_out );
  }
}
