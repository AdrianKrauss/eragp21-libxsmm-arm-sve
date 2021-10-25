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

#include "generator_packed_gemm_bc_rm_aarch64_sve.h"
#include "generator_aarch64_instructions.h"
#include "generator_common_aarch64.h"
#include "generator_gemm_common_aarch64.h"
#include "libxsmm_main.h"

LIBXSMM_API_INTERN
void libxsmm_generator_packed_gemm_bc_rm_aarch64_sve(   libxsmm_generated_code*         io_generated_code,
                                                        const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                                        const unsigned int              i_packed_width ) {
  unsigned int l_max_reg_block = 0;
  unsigned int l_n1_range = 0;
  unsigned int l_n2_range = 0;
  unsigned int l_n1_block = 0;
  unsigned int l_n2_block = 0;

  libxsmm_micro_kernel_config l_micro_kernel_config;
  libxsmm_loop_label_tracker l_loop_label_tracker;
  libxsmm_gp_reg_mapping l_gp_reg_mapping;

  /* select accumulator blocking */
  l_max_reg_block = 30;

  /* define gp register mapping */
  libxsmm_reset_aarch64_gp_reg_mapping( &l_gp_reg_mapping );

  l_gp_reg_mapping.gp_reg_a = LIBXSMM_AARCH64_GP_REG_X0;
  l_gp_reg_mapping.gp_reg_b = LIBXSMM_AARCH64_GP_REG_X1;
  l_gp_reg_mapping.gp_reg_c = LIBXSMM_AARCH64_GP_REG_X2;
  l_gp_reg_mapping.gp_reg_a_prefetch = LIBXSMM_AARCH64_GP_REG_X3;
  l_gp_reg_mapping.gp_reg_b_prefetch = LIBXSMM_AARCH64_GP_REG_X4;
  /*l_gp_reg_mapping.gp_reg_c_prefetch = LIBXSMM_AARCH64_GP_REG_X5;*/
  l_gp_reg_mapping.gp_reg_mloop = LIBXSMM_AARCH64_GP_REG_X6;
  l_gp_reg_mapping.gp_reg_nloop = LIBXSMM_AARCH64_GP_REG_X7;
  l_gp_reg_mapping.gp_reg_kloop = LIBXSMM_AARCH64_GP_REG_X8;
  l_gp_reg_mapping.gp_reg_help_0 = LIBXSMM_AARCH64_GP_REG_X9;
  l_gp_reg_mapping.gp_reg_help_1 = LIBXSMM_AARCH64_GP_REG_X10;
  l_gp_reg_mapping.gp_reg_help_2 = LIBXSMM_AARCH64_GP_REG_X11;
  l_gp_reg_mapping.gp_reg_help_3 = LIBXSMM_AARCH64_GP_REG_X12;     /* this is the SVE packed register loop */
  l_gp_reg_mapping.gp_reg_help_4 = LIBXSMM_AARCH64_GP_REG_UNDEF;
  l_gp_reg_mapping.gp_reg_help_5 = LIBXSMM_AARCH64_GP_REG_UNDEF;

  /* define loop_label_tracker */
  libxsmm_reset_loop_label_tracker( &l_loop_label_tracker );

  /* define the micro kernel code gen properties */
  libxsmm_generator_gemm_init_micro_kernel_config_aarch64( &l_micro_kernel_config, io_generated_code->arch, i_xgemm_desc );

  /* calculate the chunk size of current columns to work on */
  if ( libxsmm_compute_equalized_blocking( i_xgemm_desc->n, l_max_reg_block, &l_n1_range, &l_n1_block, &l_n2_range, &l_n2_block ) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_N_BLOCK );
    return;
  }

  /* open asm */
  libxsmm_aarch64_instruction_open_stream( io_generated_code, 0xf );

  /* set p0 to all true using ptrue */
  libxsmm_generator_set_p_register_aarch64_sve( io_generated_code, LIBXSMM_AARCH64_SVE_REG_P0, 
                                                -1, LIBXSMM_AARCH64_SVE_REG_UNDEF );

  /* m loop */
  libxsmm_generator_loop_header_aarch64( io_generated_code, &l_loop_label_tracker, l_gp_reg_mapping.gp_reg_mloop, i_xgemm_desc->m );

  /* loop over n-blocks */
  if ( l_n1_block == i_xgemm_desc->n ) {
    /* no N loop at all */
    libxsmm_generator_packed_gemm_bc_rm_aarch64_kloop_sve( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_micro_kernel_config, i_xgemm_desc,
                                                           i_packed_width, i_xgemm_desc->n );
  } else if ( (l_n1_range > 0) && (l_n2_range > 0) ) {
    /* we have two ranges */
    /* first range */
    libxsmm_generator_loop_header_aarch64( io_generated_code, &l_loop_label_tracker, l_gp_reg_mapping.gp_reg_nloop, l_n1_range );

    libxsmm_generator_packed_gemm_bc_rm_aarch64_kloop_sve( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_micro_kernel_config, i_xgemm_desc,
                                                           i_packed_width, l_n1_block );

    libxsmm_generator_loop_footer_aarch64( io_generated_code, &l_loop_label_tracker, l_gp_reg_mapping.gp_reg_nloop, l_n1_block );

    /* second range */
    libxsmm_generator_loop_header_aarch64( io_generated_code, &l_loop_label_tracker, l_gp_reg_mapping.gp_reg_nloop, i_xgemm_desc->n - l_n1_range );

    libxsmm_generator_packed_gemm_bc_rm_aarch64_kloop_sve( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_micro_kernel_config, i_xgemm_desc,
                                                           i_packed_width, l_n2_block );

    libxsmm_generator_loop_footer_aarch64( io_generated_code, &l_loop_label_tracker, l_gp_reg_mapping.gp_reg_nloop, l_n2_block );

    /* reset B and C pointer */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   l_gp_reg_mapping.gp_reg_b, l_gp_reg_mapping.gp_reg_help_1, l_gp_reg_mapping.gp_reg_b,
                                                   i_xgemm_desc->n * i_packed_width * l_micro_kernel_config.datatype_size_in );
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   l_gp_reg_mapping.gp_reg_c, l_gp_reg_mapping.gp_reg_help_2, l_gp_reg_mapping.gp_reg_c,
                                                   i_xgemm_desc->n * i_packed_width * l_micro_kernel_config.datatype_size_out );
  } else if ( (l_n1_range > 0) && (l_n2_range == 0) ) {
    /* reset n loop */
    /* we have only one range */
    libxsmm_generator_loop_header_aarch64( io_generated_code, &l_loop_label_tracker, l_gp_reg_mapping.gp_reg_nloop, i_xgemm_desc->n );

    libxsmm_generator_packed_gemm_bc_rm_aarch64_kloop_sve( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_micro_kernel_config, i_xgemm_desc,
                                                           i_packed_width, l_n1_block );

    libxsmm_generator_loop_footer_aarch64( io_generated_code, &l_loop_label_tracker, l_gp_reg_mapping.gp_reg_nloop, l_n1_block );

    /* reset B and C pointer */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   l_gp_reg_mapping.gp_reg_b, l_gp_reg_mapping.gp_reg_help_1, l_gp_reg_mapping.gp_reg_b,
                                                   i_xgemm_desc->n * i_packed_width * l_micro_kernel_config.datatype_size_in );
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   l_gp_reg_mapping.gp_reg_c, l_gp_reg_mapping.gp_reg_help_2, l_gp_reg_mapping.gp_reg_c,
                                                   i_xgemm_desc->n * i_packed_width * l_micro_kernel_config.datatype_size_out );
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_N_BLOCK );
    return;
  }

  /* advance A pointer */
  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                 l_gp_reg_mapping.gp_reg_a, l_gp_reg_mapping.gp_reg_help_0, l_gp_reg_mapping.gp_reg_a,
                                                 l_micro_kernel_config.datatype_size_in*i_xgemm_desc->lda );
  /* advance C pointer */
  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                 l_gp_reg_mapping.gp_reg_c, l_gp_reg_mapping.gp_reg_help_2, l_gp_reg_mapping.gp_reg_c,
                                                 l_micro_kernel_config.datatype_size_out*i_packed_width*i_xgemm_desc->ldc );

  /* close m loop */
  libxsmm_generator_loop_footer_aarch64( io_generated_code, &l_loop_label_tracker, l_gp_reg_mapping.gp_reg_mloop, 1 );

  /* close asm */
  libxsmm_aarch64_instruction_close_stream( io_generated_code, 0xf );
}

LIBXSMM_API_INTERN
void libxsmm_generator_packed_gemm_bc_rm_aarch64_kloop_sve( libxsmm_generated_code*            io_generated_code,
                                                            libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                                            const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                            const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                            const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                            const unsigned int                 i_packed_width,
                                                            const unsigned int                 i_n_blocking ) {
  /* calculate how many iterations we need */

  unsigned int l_sve_packed_remainder = 0;
  unsigned int l_sve_packed_iters = 0;
  unsigned int l_sve_packed_width = 0;

  l_sve_packed_width = i_micro_kernel_config->vector_length;
  l_sve_packed_remainder = i_packed_width % l_sve_packed_width;
  l_sve_packed_iters = i_packed_width/l_sve_packed_width;

  /* check if we have a single SVE devisor */
  if ( l_sve_packed_width == i_packed_width ) {
    /* run inner compute kernel with p0 once */
    libxsmm_generator_packed_gemm_bc_rm_aarch64_kloop_sve_packed( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc,
                                                                  i_packed_width, 0, i_n_blocking, LIBXSMM_AARCH64_SVE_REG_P0 );
  /* check if we have a perfect SVE devisor */
  } else if ( l_sve_packed_remainder == 0 ) {
    /* initialize packed loop */
    libxsmm_generator_loop_header_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_help_3, i_packed_width );

    /* run inner compute kernel with p0 */
    libxsmm_generator_packed_gemm_bc_rm_aarch64_kloop_sve_packed( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc,
                                                                  i_packed_width, 0, i_n_blocking, LIBXSMM_AARCH64_SVE_REG_P0 );

    /* advance B pointer */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   i_gp_reg_mapping->gp_reg_b, i_gp_reg_mapping->gp_reg_help_1, i_gp_reg_mapping->gp_reg_b,
                                                   l_sve_packed_width * i_micro_kernel_config->datatype_size_in );

    /* advance C pointer */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   i_gp_reg_mapping->gp_reg_c, i_gp_reg_mapping->gp_reg_help_2, i_gp_reg_mapping->gp_reg_c,
                                                   l_sve_packed_width * i_micro_kernel_config->datatype_size_out );

    /* jump back to pack loop label */
    libxsmm_generator_loop_footer_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_help_3, l_sve_packed_width );

    /* reset B pointer */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   i_gp_reg_mapping->gp_reg_b, i_gp_reg_mapping->gp_reg_help_1, i_gp_reg_mapping->gp_reg_b,
                                                   l_sve_packed_iters * l_sve_packed_width * i_micro_kernel_config->datatype_size_in );

    /* reset C pointer */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   i_gp_reg_mapping->gp_reg_c, i_gp_reg_mapping->gp_reg_help_2, i_gp_reg_mapping->gp_reg_c,
                                                   l_sve_packed_iters * l_sve_packed_width * i_micro_kernel_config->datatype_size_out );

  /* we have less than SVE width and apply masking with p1 */
  } else if ( l_sve_packed_width > i_packed_width  ) {
    /* set p1 to true for #remainder elements using whilelt */
    libxsmm_generator_set_p_register_aarch64_sve( io_generated_code, LIBXSMM_AARCH64_SVE_REG_P1, 
                                                  l_sve_packed_remainder*i_micro_kernel_config->datatype_size_in, i_gp_reg_mapping->gp_reg_help_1 );

     /* run inner compute kernel for remainder with p1 once */
    libxsmm_generator_packed_gemm_bc_rm_aarch64_kloop_sve_packed( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc,
                                                                  i_packed_width, l_sve_packed_remainder, i_n_blocking, LIBXSMM_AARCH64_SVE_REG_P1 );  
  /* the general case */
  } else {
    /* initialize packed loop for full packed iterations */
    libxsmm_generator_loop_header_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_help_3, l_sve_packed_iters );

    /* run inner compute kernel with p0 */
    libxsmm_generator_packed_gemm_bc_rm_aarch64_kloop_sve_packed( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc,
                                                                  i_packed_width, 0, i_n_blocking, LIBXSMM_AARCH64_SVE_REG_P0 );

    /* advance B pointer */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   i_gp_reg_mapping->gp_reg_b, i_gp_reg_mapping->gp_reg_help_1, i_gp_reg_mapping->gp_reg_b,
                                                   l_sve_packed_width * i_micro_kernel_config->datatype_size_in );

    /* advance C pointer */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   i_gp_reg_mapping->gp_reg_c, i_gp_reg_mapping->gp_reg_help_2, i_gp_reg_mapping->gp_reg_c,
                                                   l_sve_packed_width * i_micro_kernel_config->datatype_size_out );

    /* jump back to pack loop label */
    libxsmm_generator_loop_footer_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_help_3, 1 );

    /* set p1 to true for #remainder elements using whilelt */
    libxsmm_generator_set_p_register_aarch64_sve( io_generated_code, LIBXSMM_AARCH64_SVE_REG_P1, 
                                                  l_sve_packed_remainder*i_micro_kernel_config->datatype_size_in, i_gp_reg_mapping->gp_reg_help_1 );

    /* run inner compute kernel for remainder with p1 once */
    libxsmm_generator_packed_gemm_bc_rm_aarch64_kloop_sve_packed( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc,
                                                                  i_packed_width, l_sve_packed_remainder, i_n_blocking, LIBXSMM_AARCH64_SVE_REG_P1 );

    /* reset B pointer */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   i_gp_reg_mapping->gp_reg_b, i_gp_reg_mapping->gp_reg_help_1, i_gp_reg_mapping->gp_reg_b,
                                                   l_sve_packed_iters * l_sve_packed_width * i_micro_kernel_config->datatype_size_in );

    /* reset C pointer */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   i_gp_reg_mapping->gp_reg_c, i_gp_reg_mapping->gp_reg_help_2, i_gp_reg_mapping->gp_reg_c,
                                                   l_sve_packed_iters * l_sve_packed_width * i_micro_kernel_config->datatype_size_out );
  }

  /* advance B and C pointers if N is bigger than our register blocking */
  if ( i_xgemm_desc->n != i_n_blocking ) {
    /* advance B pointer */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   i_gp_reg_mapping->gp_reg_b, i_gp_reg_mapping->gp_reg_help_1, i_gp_reg_mapping->gp_reg_b,
                                                   i_n_blocking * i_packed_width * i_micro_kernel_config->datatype_size_in );

    /* advance C pointer */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   i_gp_reg_mapping->gp_reg_c, i_gp_reg_mapping->gp_reg_help_2, i_gp_reg_mapping->gp_reg_c,
                                                   i_n_blocking * i_packed_width * i_micro_kernel_config->datatype_size_out );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_packed_gemm_bc_rm_aarch64_kloop_sve_packed( libxsmm_generated_code*            io_generated_code,
                                                                   libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                                                   const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                                   const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                                   const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                                   const unsigned int                 i_packed_width,
                                                                   const unsigned int                 i_sve_packed_remainder,
                                                                   const unsigned int                 i_n_blocking, 
                                                                   const unsigned char                i_pred_reg ) {
  unsigned int l_n = 0;
  unsigned int l_use_masking = 0;

  /* check if we need masking */
  if ( i_sve_packed_remainder != 0 ) {
    l_use_masking = 1;
  }

  /* load C accumulator */
  for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
    if (0 != (LIBXSMM_GEMM_FLAG_BETA_0 & i_xgemm_desc->flags)) { /* Beta=0 */
      libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_EOR_V,
                                               l_n, l_n, 0, l_n, LIBXSMM_AARCH64_SVE_REG_UNDEF, 
                                               (i_micro_kernel_config->datatype_size_in == 4) ? LIBXSMM_AARCH64_SVE_TYPE_S : LIBXSMM_AARCH64_SVE_TYPE_D );
    } else {
      /* if masking is set, load with LD1W / LD1D and predicate to avoid segmentation fault */
      if ( l_use_masking ) {
        /* load C values with masking predicate */
        libxsmm_aarch64_instruction_sve_move( io_generated_code, (i_micro_kernel_config->datatype_size_in == 4) ? LIBXSMM_AARCH64_INSTR_SVE_LD1W_I_OFF : LIBXSMM_AARCH64_INSTR_SVE_LD1D_I_OFF, 
                                              i_gp_reg_mapping->gp_reg_c, LIBXSMM_AARCH64_GP_REG_UNDEF,
                                              0, l_n, i_pred_reg );

        /* advance C pointer */
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                       i_gp_reg_mapping->gp_reg_c, i_gp_reg_mapping->gp_reg_help_2, i_gp_reg_mapping->gp_reg_c,
                                                       i_packed_width*i_micro_kernel_config->datatype_size_out );
      /* if no masking needed, LDR is used */
      } else {
        /* load C values without masking predicate */
        libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_LDR_Z_I_OFF, 
                                              i_gp_reg_mapping->gp_reg_c, LIBXSMM_AARCH64_GP_REG_UNDEF,
                                              0, l_n, LIBXSMM_AARCH64_SVE_REG_UNDEF );

        /* advance C pointer */
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                       i_gp_reg_mapping->gp_reg_c, i_gp_reg_mapping->gp_reg_help_2, i_gp_reg_mapping->gp_reg_c,
                                                       i_packed_width*i_micro_kernel_config->datatype_size_out );                                       
      }
    }
  }
  /* reset C point for stores */
  if (0 == (LIBXSMM_GEMM_FLAG_BETA_0 & i_xgemm_desc->flags)) {
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   i_gp_reg_mapping->gp_reg_c, i_gp_reg_mapping->gp_reg_help_2, i_gp_reg_mapping->gp_reg_c,
                                                   i_n_blocking*i_packed_width*i_micro_kernel_config->datatype_size_out );
  }
  /* k loop */
  libxsmm_generator_loop_header_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_kloop, i_xgemm_desc->k );

  /* broadcast of A */
  libxsmm_aarch64_instruction_sve_move( io_generated_code, (i_micro_kernel_config->datatype_size_in == 4) ? LIBXSMM_AARCH64_INSTR_SVE_LD1RW_I_OFF : LIBXSMM_AARCH64_INSTR_SVE_LD1RD_I_OFF,
                                        i_gp_reg_mapping->gp_reg_a, LIBXSMM_AARCH64_GP_REG_UNDEF,
                                        0, i_n_blocking, i_pred_reg );

  /* advance A pointer */
  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                 i_gp_reg_mapping->gp_reg_a, i_gp_reg_mapping->gp_reg_help_2, i_gp_reg_mapping->gp_reg_a,
                                                 i_micro_kernel_config->datatype_size_in );

  /* loop over the register block */
  for ( l_n = 0; l_n < i_n_blocking; ++l_n ) {
    /* load B with predicate*/
    libxsmm_aarch64_instruction_sve_move( io_generated_code, (i_micro_kernel_config->datatype_size_in == 4) ? LIBXSMM_AARCH64_INSTR_SVE_LD1W_I_OFF : LIBXSMM_AARCH64_INSTR_SVE_LD1D_I_OFF, 
                                          i_gp_reg_mapping->gp_reg_b, LIBXSMM_AARCH64_GP_REG_UNDEF,
                                          0, i_n_blocking+1, i_pred_reg );

    /* advance B pointer */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   i_gp_reg_mapping->gp_reg_b, i_gp_reg_mapping->gp_reg_help_2, i_gp_reg_mapping->gp_reg_b,
                                                   i_packed_width * i_micro_kernel_config->datatype_size_in );

    /* FMLA */
    libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FMLA_V,
                                             i_n_blocking+1, i_n_blocking, 0, l_n,
                                             i_pred_reg, (i_micro_kernel_config->datatype_size_in == 4) ? LIBXSMM_AARCH64_SVE_TYPE_S : LIBXSMM_AARCH64_SVE_TYPE_D );  
  }
  /* advance B pointer */
  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                 i_gp_reg_mapping->gp_reg_b, i_gp_reg_mapping->gp_reg_help_1, i_gp_reg_mapping->gp_reg_b,
                                                 (unsigned long long)((i_packed_width * i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size_in)
                                                 - (i_n_blocking * i_packed_width * i_micro_kernel_config->datatype_size_in)) );

  /* close k loop */
  libxsmm_generator_loop_footer_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_kloop, 1 );

  /* store C accumulator */
  for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
    /* if masking is set, store with ST1W / ST1D and predicate to avoid segmentation fault */
    if ( l_use_masking ) {
      /* store C values with predicate */
      libxsmm_aarch64_instruction_sve_move( io_generated_code, (i_micro_kernel_config->datatype_size_in == 4) ? LIBXSMM_AARCH64_INSTR_SVE_ST1W_I_OFF : LIBXSMM_AARCH64_INSTR_SVE_ST1D_I_OFF, 
                                            i_gp_reg_mapping->gp_reg_c, LIBXSMM_AARCH64_GP_REG_UNDEF,
                                            0, l_n, i_pred_reg );

      /* advance C pointer */
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                     i_gp_reg_mapping->gp_reg_c, i_gp_reg_mapping->gp_reg_help_2, i_gp_reg_mapping->gp_reg_c,
                                                     i_packed_width*i_micro_kernel_config->datatype_size_out );
    /* if no masking needed, store with STR */
    } else {
      /* store C values without masking predicate */
      libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_STR_Z_I_OFF, 
                                            i_gp_reg_mapping->gp_reg_c, LIBXSMM_AARCH64_GP_REG_UNDEF,
                                            0, l_n, LIBXSMM_AARCH64_SVE_REG_UNDEF );

      /* advance C pointer */
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                     i_gp_reg_mapping->gp_reg_c, i_gp_reg_mapping->gp_reg_help_2, i_gp_reg_mapping->gp_reg_c,
                                                     i_packed_width*i_micro_kernel_config->datatype_size_out );
    }
  }
  /* reset C pointer */
  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                 i_gp_reg_mapping->gp_reg_c, i_gp_reg_mapping->gp_reg_help_2, i_gp_reg_mapping->gp_reg_c,
                                                 i_n_blocking*i_packed_width*i_micro_kernel_config->datatype_size_out );

  /* reset A pointer */
  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                 i_gp_reg_mapping->gp_reg_a, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_a,
                                                 (unsigned long long)(i_xgemm_desc->k*i_micro_kernel_config->datatype_size_in) );

  /* reset B pointer */
  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                 i_gp_reg_mapping->gp_reg_b, i_gp_reg_mapping->gp_reg_help_1, i_gp_reg_mapping->gp_reg_b,
                                                 (unsigned long long)(i_xgemm_desc->k * i_xgemm_desc->ldb * i_packed_width * i_micro_kernel_config->datatype_size_in) );
}

