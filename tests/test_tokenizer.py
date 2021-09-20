import unittest
from tokenizecode import CodeTokenizer, CodeParser, TreeBPE, SplitLinesTraversal
from tokenizecode.bpe import WordPieceBPE

SAMPLE_CODE = {
    'agda': """------------------------------------------------------------------------
-- The Agda standard library
--
-- This module is DEPRECATED.
------------------------------------------------------------------------

{-# OPTIONS --without-K --safe #-}

open import Algebra

module Algebra.Operations.Ring
  {ℓ₁ ℓ₂} (ring : RawRing ℓ₁ ℓ₂)
  where

{-# WARNING_ON_IMPORT
"Algebra.Operations.Ring was deprecated in v1.5.
Use Algebra.Properties.Semiring.Exp(.TCOptimised) instead."
#-}

open import Data.Nat.Base as ℕ using (ℕ; suc; zero)

open RawRing ring

infixr 8 _^_+1
_^_+1 : Carrier → ℕ → Carrier
x ^ zero  +1 = x
x ^ suc n +1 = (x ^ n +1) * x

infixr 8 _^_
_^_ : Carrier → ℕ → Carrier
x ^ zero  = 1#
x ^ suc i = x ^ i +1
{-# INLINE _^_ #-}
""",
    'c': """/**
 * Author......: See docs/credits.txt
 * License.....: MIT
 */

#include "common.h"
#include "types.h"
#include "event.h"
#include "backend.h"
#include "status.h"
#include "autotune.h"

static double try_run (hashcat_ctx_t *hashcat_ctx, hc_device_param_t *device_param, const u32 kernel_accel, const u32 kernel_loops, const u32 kernel_threads)
{
  hashconfig_t   *hashconfig   = hashcat_ctx->hashconfig;
  user_options_t *user_options = hashcat_ctx->user_options;

  device_param->kernel_params_buf32[28] = 0;
  device_param->kernel_params_buf32[29] = kernel_loops; // not a bug, both need to be set
  device_param->kernel_params_buf32[30] = kernel_loops; // because there's two variables for inner iters for slow and fast hashes

  const u32 hardware_power = ((hashconfig->opts_type & OPTS_TYPE_MP_MULTI_DISABLE) ? 1 : device_param->device_processors) * kernel_threads;

  u32 kernel_power_try = hardware_power * kernel_accel;

  if (user_options->attack_mode == ATTACK_MODE_ASSOCIATION)
  {
    hashes_t *hashes = hashcat_ctx->hashes;

    const u32 salts_cnt = hashes->salts_cnt;

    if (kernel_power_try > salts_cnt)
    {
      kernel_power_try = salts_cnt;
    }
  }

  const u32 kernel_threads_sav = device_param->kernel_threads;

  device_param->kernel_threads = kernel_threads;

  const double spin_damp_sav = device_param->spin_damp;

  device_param->spin_damp = 0;

  if (hashconfig->attack_exec == ATTACK_EXEC_INSIDE_KERNEL)
  {
    if (hashconfig->opti_type & OPTI_TYPE_OPTIMIZED_KERNEL)
    {
      run_kernel (hashcat_ctx, device_param, KERN_RUN_1, 0, kernel_power_try, true, 0);
    }
    else
    {
      run_kernel (hashcat_ctx, device_param, KERN_RUN_4, 0, kernel_power_try, true, 0);
    }
  }
  else
  {
    run_kernel (hashcat_ctx, device_param, KERN_RUN_2, 0, kernel_power_try, true, 0);
  }

  device_param->spin_damp = spin_damp_sav;

  device_param->kernel_threads = kernel_threads_sav;

  const double exec_msec_prev = get_avg_exec_time (device_param, 1);

  return exec_msec_prev;
}

static double try_run_times (hashcat_ctx_t *hashcat_ctx, hc_device_param_t *device_param, const u32 kernel_accel, const u32 kernel_loops, const u32 kernel_threads, const int times)
{
  double exec_msec_best = try_run (hashcat_ctx, device_param, kernel_accel, kernel_loops, kernel_threads);

  for (int i = 1; i < times; i++)
  {
    double exec_msec = try_run (hashcat_ctx, device_param, kernel_accel, kernel_loops, kernel_threads);

    if (exec_msec > exec_msec_best) continue;

    exec_msec_best = exec_msec;
  }

  return exec_msec_best;
}

static u32 previous_power_of_two (const u32 x)
{
  // https://stackoverflow.com/questions/2679815/previous-power-of-2
  // really cool!

  if (x == 0) return 0;

  u32 r = x;

  r |= (r >>  1);
  r |= (r >>  2);
  r |= (r >>  4);
  r |= (r >>  8);
  r |= (r >> 16);

  return r - (r >> 1);
}

static int autotune (hashcat_ctx_t *hashcat_ctx, hc_device_param_t *device_param)
{
  const hashconfig_t    *hashconfig   = hashcat_ctx->hashconfig;
  const backend_ctx_t   *backend_ctx  = hashcat_ctx->backend_ctx;
  const straight_ctx_t  *straight_ctx = hashcat_ctx->straight_ctx;
  const user_options_t  *user_options = hashcat_ctx->user_options;

  const double target_msec = backend_ctx->target_msec;

  const u32 kernel_accel_min = device_param->kernel_accel_min;
  const u32 kernel_accel_max = device_param->kernel_accel_max;

  const u32 kernel_loops_min = device_param->kernel_loops_min;
  const u32 kernel_loops_max = device_param->kernel_loops_max;

  const u32 kernel_threads_min = device_param->kernel_threads_min;
  const u32 kernel_threads_max = device_param->kernel_threads_max;

  u32 kernel_accel = kernel_accel_min;
  u32 kernel_loops = kernel_loops_min;

  // for the threads we take as initial value what we receive from the runtime
  // but is only to start with something, we will fine tune this value as soon as we have our workload specified
  // this thread limiting is also performed insinde run_kernel() so we need to redo it here, too

  u32 kernel_wgs = 0;
  u32 kernel_wgs_multiple = 0;

  if (hashconfig->attack_exec == ATTACK_EXEC_INSIDE_KERNEL)
  {
    if (hashconfig->opti_type & OPTI_TYPE_OPTIMIZED_KERNEL)
    {
      kernel_wgs = device_param->kernel_wgs1;

      kernel_wgs_multiple = device_param->kernel_preferred_wgs_multiple1;
    }
    else
    {
      kernel_wgs = device_param->kernel_wgs4;

      kernel_wgs_multiple = device_param->kernel_preferred_wgs_multiple4;
    }
  }
  else
  {
    kernel_wgs = device_param->kernel_wgs2;

    kernel_wgs_multiple = device_param->kernel_preferred_wgs_multiple2;
  }

  u32 kernel_threads = kernel_threads_max;

  if ((kernel_wgs >= kernel_threads_min) && (kernel_wgs <= kernel_threads_max))
  {
    kernel_threads = kernel_wgs;
  }

  // having a value power of 2 makes it easier to divide

  const u32 kernel_threads_p2 = previous_power_of_two (kernel_threads);

  if ((kernel_threads_p2 >= kernel_threads_min) && (kernel_threads_p2 <= kernel_threads_max))
  {
    kernel_threads = kernel_threads_p2;
  }

  // in this case the user specified a fixed -n and -u on the commandline
  // no way to tune anything
  // but we need to run a few caching rounds

  if ((kernel_accel_min == kernel_accel_max) && (kernel_loops_min == kernel_loops_max))
  {
    #if defined (DEBUG)

    // don't do any autotune in debug mode in this case
    // we're propably during kernel development

    #else

    if (hashconfig->warmup_disable == false)
    {
      try_run (hashcat_ctx, device_param, kernel_accel, kernel_loops, kernel_threads);
      try_run (hashcat_ctx, device_param, kernel_accel, kernel_loops, kernel_threads);
      try_run (hashcat_ctx, device_param, kernel_accel, kernel_loops, kernel_threads);
      try_run (hashcat_ctx, device_param, kernel_accel, kernel_loops, kernel_threads);
    }

    #endif
  }
  else
  {
    // from here it's clear we are allowed to autotune
    // so let's init some fake words

    const u32 hardware_power_max = ((hashconfig->opts_type & OPTS_TYPE_MP_MULTI_DISABLE) ? 1 : device_param->device_processors) * kernel_threads_max;

    u32 kernel_power_max = hardware_power_max * kernel_accel_max;

    if (user_options->attack_mode == ATTACK_MODE_ASSOCIATION)
    {
      hashes_t *hashes = hashcat_ctx->hashes;

      const u32 salts_cnt = hashes->salts_cnt;

      if (kernel_power_max > salts_cnt)
      {
        kernel_power_max = salts_cnt;
      }
    }

    if (device_param->is_cuda == true)
    {
      if (run_cuda_kernel_atinit (hashcat_ctx, device_param, device_param->cuda_d_pws_buf, kernel_power_max) == -1) return -1;
    }

    if (device_param->is_hip == true)
    {
      if (run_hip_kernel_atinit (hashcat_ctx, device_param, device_param->hip_d_pws_buf, kernel_power_max) == -1) return -1;
    }

    if (device_param->is_opencl == true)
    {
      if (run_opencl_kernel_atinit (hashcat_ctx, device_param, device_param->opencl_d_pws_buf, kernel_power_max) == -1) return -1;
    }

    if (user_options->slow_candidates == true)
    {
    }
    else
    {
      if (hashconfig->attack_exec == ATTACK_EXEC_INSIDE_KERNEL)
      {
        if (straight_ctx->kernel_rules_cnt > 1)
        {
          if (device_param->is_cuda == true)
          {
            if (hc_cuMemcpyDtoDAsync (hashcat_ctx, device_param->cuda_d_rules_c, device_param->cuda_d_rules, MIN (kernel_loops_max, KERNEL_RULES) * sizeof (kernel_rule_t), device_param->cuda_stream) == -1) return -1;
          }

          if (device_param->is_hip == true)
          {
            if (hc_hipMemcpyDtoDAsync (hashcat_ctx, device_param->hip_d_rules_c, device_param->hip_d_rules, MIN (kernel_loops_max, KERNEL_RULES) * sizeof (kernel_rule_t), device_param->hip_stream) == -1) return -1;
          }

          if (device_param->is_opencl == true)
          {
            if (hc_clEnqueueCopyBuffer (hashcat_ctx, device_param->opencl_command_queue, device_param->opencl_d_rules, device_param->opencl_d_rules_c, 0, 0, MIN (kernel_loops_max, KERNEL_RULES) * sizeof (kernel_rule_t), 0, NULL, NULL) == -1) return -1;
          }
        }
      }
    }

    // we also need to initialize some values using kernels

    if (hashconfig->attack_exec == ATTACK_EXEC_INSIDE_KERNEL)
    {
      // nothing to do
    }
    else
    {
      const u32 kernel_threads_sav = device_param->kernel_threads;

      device_param->kernel_threads = device_param->kernel_wgs1;

      run_kernel (hashcat_ctx, device_param, KERN_RUN_1, 0, kernel_power_max, false, 0);

      if (hashconfig->opts_type & OPTS_TYPE_LOOP_PREPARE)
      {
        device_param->kernel_threads = device_param->kernel_wgs2p;

        run_kernel (hashcat_ctx, device_param, KERN_RUN_2P, 0, kernel_power_max, false, 0);
      }

      device_param->kernel_threads = kernel_threads_sav;
    }

    // Do a pre-autotune test run to find out if kernel runtime is above some TDR limit

    u32 kernel_loops_max_reduced = kernel_loops_max;

    if (true)
    {
      double exec_msec = try_run (hashcat_ctx, device_param, kernel_accel_min, kernel_loops_min, kernel_threads);

      if (exec_msec > 2000)
      {
        event_log_error (hashcat_ctx, "Kernel minimum runtime larger than default TDR");

        return -1;
      }

      exec_msec = try_run (hashcat_ctx, device_param, kernel_accel_min, kernel_loops_min, kernel_threads);

      const u32 mm = kernel_loops_max / kernel_loops_min;

      if ((exec_msec * mm) > target_msec)
      {
        const u32 loops_valid = (const u32) (target_msec / exec_msec);

        kernel_loops_max_reduced = kernel_loops_min * loops_valid;
      }
    }

    // first find out highest kernel-loops that stays below target_msec

    if (kernel_loops_min < kernel_loops_max)
    {
      for (kernel_loops = kernel_loops_max; kernel_loops > kernel_loops_min; kernel_loops >>= 1)
      {
        if (kernel_loops > kernel_loops_max_reduced) continue;

        double exec_msec = try_run_times (hashcat_ctx, device_param, kernel_accel_min, kernel_loops, kernel_threads, 1);

        if (exec_msec < target_msec) break;
      }
    }

    #define STEPS_CNT 16

    // now the same for kernel-accel but with the new kernel-loops from previous loop set

    if (kernel_accel_min < kernel_accel_max)
    {
      for (int i = 0; i < STEPS_CNT; i++)
      {
        const u32 kernel_accel_try = 1U << i;

        if (kernel_accel_try < kernel_accel_min) continue;
        if (kernel_accel_try > kernel_accel_max) break;

        double exec_msec = try_run_times (hashcat_ctx, device_param, kernel_accel_try, kernel_loops, kernel_threads, 1);

        if (exec_msec > target_msec) break;

        kernel_accel = kernel_accel_try;
      }
    }

    // now find the middle balance between kernel_accel and kernel_loops
    // while respecting allowed ranges at the same time

    if (kernel_accel < kernel_loops)
    {
      const u32 kernel_accel_orig = kernel_accel;
      const u32 kernel_loops_orig = kernel_loops;

      double exec_msec_prev = try_run_times (hashcat_ctx, device_param, kernel_accel, kernel_loops, kernel_threads, 1);

      for (int i = 1; i < STEPS_CNT; i++)
      {
        const u32 kernel_accel_try = kernel_accel_orig * (1U << i);
        const u32 kernel_loops_try = kernel_loops_orig / (1U << i);

        if (kernel_accel_try < kernel_accel_min) continue;
        if (kernel_accel_try > kernel_accel_max) break;

        if (kernel_loops_try > kernel_loops_max) continue;
        if (kernel_loops_try < kernel_loops_min) break;

        // do a real test

        const double exec_msec = try_run_times (hashcat_ctx, device_param, kernel_accel_try, kernel_loops_try, kernel_threads, 1);

        if (exec_msec_prev < exec_msec) break;

        exec_msec_prev = exec_msec;

        // so far, so good! save

        kernel_accel = kernel_accel_try;
        kernel_loops = kernel_loops_try;

        // too much if the next test is true

        if (kernel_loops_try < kernel_accel_try) break;
      }
    }

    double exec_msec_pre_final = try_run_times (hashcat_ctx, device_param, kernel_accel, kernel_loops, kernel_threads, 1);

    const u32 exec_left = (const u32) (target_msec / exec_msec_pre_final);

    const u32 accel_left = kernel_accel_max / kernel_accel;

    const u32 exec_accel_min = MIN (exec_left, accel_left); // we want that to be int

    if (exec_accel_min >= 1)
    {
      // this is safe to not overflow kernel_accel_max because of accel_left

      kernel_accel *= exec_accel_min;
    }

    // v6.2.4 new section: find thread count
    // This is not as effective as it could be because of inaccurate kernel return timers
    // But is better than fixed values
    // Timers in this section are critical, so we rerun meassurements 3 times

    if (kernel_threads_max > kernel_threads_min)
    {
      const u32 kernel_accel_orig   = kernel_accel;
      const u32 kernel_threads_orig = kernel_threads;

      double exec_msec_prev = try_run_times (hashcat_ctx, device_param, kernel_accel, kernel_loops, kernel_threads, 3);

      for (int i = 1; i < STEPS_CNT; i++)
      {
        const u32 kernel_accel_try   = kernel_accel_orig   * (1U << i);
        const u32 kernel_threads_try = kernel_threads_orig / (1U << i);

        // since we do not modify total amount of workitems, we can (and need) to do increase kernel_accel_max

        const u32 kernel_accel_max_try = kernel_accel_max * (1U << i);

        if (kernel_accel_try > kernel_accel_max_try) break;

        if (kernel_threads_try < kernel_threads_min) break;

        if (kernel_threads_try % kernel_wgs_multiple) break; // this would just be waste of time

        double exec_msec = try_run_times (hashcat_ctx, device_param, kernel_accel_try, kernel_loops, kernel_threads_try, 3);

        if (exec_msec > exec_msec_prev) continue;

        exec_msec_prev = exec_msec;

        kernel_accel   = kernel_accel_try;
        kernel_threads = kernel_threads_try;
      }
    }
  }

  // reset them fake words
  // reset other buffers in case autotune cracked something

  if (device_param->is_cuda == true)
  {
    if (run_cuda_kernel_bzero (hashcat_ctx, device_param, device_param->cuda_d_pws_buf, device_param->size_pws) == -1) return -1;

    if (run_cuda_kernel_bzero (hashcat_ctx, device_param, device_param->cuda_d_plain_bufs, device_param->size_plains) == -1) return -1;

    if (run_cuda_kernel_bzero (hashcat_ctx, device_param, device_param->cuda_d_digests_shown, device_param->size_shown) == -1) return -1;

    if (run_cuda_kernel_bzero (hashcat_ctx, device_param, device_param->cuda_d_result, device_param->size_results) == -1) return -1;

    if (run_cuda_kernel_bzero (hashcat_ctx, device_param, device_param->cuda_d_tmps, device_param->size_tmps) == -1) return -1;
  }

  if (device_param->is_hip == true)
  {
    if (run_hip_kernel_bzero (hashcat_ctx, device_param, device_param->hip_d_pws_buf, device_param->size_pws) == -1) return -1;

    if (run_hip_kernel_bzero (hashcat_ctx, device_param, device_param->hip_d_plain_bufs, device_param->size_plains) == -1) return -1;

    if (run_hip_kernel_bzero (hashcat_ctx, device_param, device_param->hip_d_digests_shown, device_param->size_shown) == -1) return -1;

    if (run_hip_kernel_bzero (hashcat_ctx, device_param, device_param->hip_d_result, device_param->size_results) == -1) return -1;

    if (run_hip_kernel_bzero (hashcat_ctx, device_param, device_param->hip_d_tmps, device_param->size_tmps) == -1) return -1;
  }

  if (device_param->is_opencl == true)
  {
    if (run_opencl_kernel_bzero (hashcat_ctx, device_param, device_param->opencl_d_pws_buf, device_param->size_pws) == -1) return -1;

    if (run_opencl_kernel_bzero (hashcat_ctx, device_param, device_param->opencl_d_plain_bufs, device_param->size_plains) == -1) return -1;

    if (run_opencl_kernel_bzero (hashcat_ctx, device_param, device_param->opencl_d_digests_shown, device_param->size_shown) == -1) return -1;

    if (run_opencl_kernel_bzero (hashcat_ctx, device_param, device_param->opencl_d_result, device_param->size_results) == -1) return -1;

    if (run_opencl_kernel_bzero (hashcat_ctx, device_param, device_param->opencl_d_tmps, device_param->size_tmps) == -1) return -1;

    if (hc_clFlush (hashcat_ctx, device_param->opencl_command_queue) == -1) return -1;
  }

  // reset timer

  device_param->exec_pos = 0;

  memset (device_param->exec_msec, 0, EXEC_CACHE * sizeof (double));

  memset (device_param->exec_us_prev1,      0, EXPECTED_ITERATIONS * sizeof (double));
  memset (device_param->exec_us_prev2,      0, EXPECTED_ITERATIONS * sizeof (double));
  memset (device_param->exec_us_prev3,      0, EXPECTED_ITERATIONS * sizeof (double));
  memset (device_param->exec_us_prev4,      0, EXPECTED_ITERATIONS * sizeof (double));
  memset (device_param->exec_us_prev_init2, 0, EXPECTED_ITERATIONS * sizeof (double));
  memset (device_param->exec_us_prev_loop2, 0, EXPECTED_ITERATIONS * sizeof (double));
  memset (device_param->exec_us_prev_aux1,  0, EXPECTED_ITERATIONS * sizeof (double));
  memset (device_param->exec_us_prev_aux2,  0, EXPECTED_ITERATIONS * sizeof (double));
  memset (device_param->exec_us_prev_aux3,  0, EXPECTED_ITERATIONS * sizeof (double));
  memset (device_param->exec_us_prev_aux4,  0, EXPECTED_ITERATIONS * sizeof (double));

  // store

  device_param->kernel_accel   = kernel_accel;
  device_param->kernel_loops   = kernel_loops;
  device_param->kernel_threads = kernel_threads;

  const u32 hardware_power = ((hashconfig->opts_type & OPTS_TYPE_MP_MULTI_DISABLE) ? 1 : device_param->device_processors) * device_param->kernel_threads;

  device_param->hardware_power = hardware_power;

  const u32 kernel_power = device_param->hardware_power * device_param->kernel_accel;

  device_param->kernel_power = kernel_power;

  return 0;
}

HC_API_CALL void *thread_autotune (void *p)
{
  thread_param_t *thread_param = (thread_param_t *) p;

  hashcat_ctx_t *hashcat_ctx = thread_param->hashcat_ctx;

  backend_ctx_t *backend_ctx = hashcat_ctx->backend_ctx;

  if (backend_ctx->enabled == false) return NULL;

  hc_device_param_t *device_param = backend_ctx->devices_param + thread_param->tid;

  if (device_param->skipped == true) return NULL;

  if (device_param->skipped_warning == true) return NULL;

  if (device_param->is_cuda == true)
  {
    if (hc_cuCtxPushCurrent (hashcat_ctx, device_param->cuda_context) == -1) return NULL;
  }

  if (device_param->is_hip == true)
  {
    if (hc_hipCtxPushCurrent (hashcat_ctx, device_param->hip_context) == -1) return NULL;
  }

  const int rc_autotune = autotune (hashcat_ctx, device_param);

  if (rc_autotune == -1)
  {
    // we should do something here, tell hashcat main that autotune failed to abort
  }

  if (device_param->is_cuda == true)
  {
    if (hc_cuCtxPopCurrent (hashcat_ctx, &device_param->cuda_context) == -1) return NULL;
  }

  if (device_param->is_hip == true)
  {
    if (hc_hipCtxPopCurrent (hashcat_ctx, &device_param->hip_context) == -1) return NULL;
  }

  return NULL;
}
""",
    'cpp': """// Copyright (c) 2017-2019 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

// Based on the public domain implementation 'merged' by D. J. Bernstein
// See https://cr.yp.to/chacha.html.

#include <crypto/common.h>
#include <crypto/chacha20.h>

#include <string.h>

constexpr static inline uint32_t rotl32(uint32_t v, int c) { return (v << c) | (v >> (32 - c)); }

#define QUARTERROUND(a,b,c,d) \
  a += b; d = rotl32(d ^ a, 16); \
  c += d; b = rotl32(b ^ c, 12); \
  a += b; d = rotl32(d ^ a, 8); \
  c += d; b = rotl32(b ^ c, 7);

static const unsigned char sigma[] = "expand 32-byte k";
static const unsigned char tau[] = "expand 16-byte k";

void ChaCha20::SetKey(const unsigned char* k, size_t keylen)
{
    const unsigned char *constants;

    input[4] = ReadLE32(k + 0);
    input[5] = ReadLE32(k + 4);
    input[6] = ReadLE32(k + 8);
    input[7] = ReadLE32(k + 12);
    if (keylen == 32) { /* recommended */
        k += 16;
        constants = sigma;
    } else { /* keylen == 16 */
        constants = tau;
    }
    input[8] = ReadLE32(k + 0);
    input[9] = ReadLE32(k + 4);
    input[10] = ReadLE32(k + 8);
    input[11] = ReadLE32(k + 12);
    input[0] = ReadLE32(constants + 0);
    input[1] = ReadLE32(constants + 4);
    input[2] = ReadLE32(constants + 8);
    input[3] = ReadLE32(constants + 12);
    input[12] = 0;
    input[13] = 0;
    input[14] = 0;
    input[15] = 0;
}

ChaCha20::ChaCha20()
{
    memset(input, 0, sizeof(input));
}

ChaCha20::ChaCha20(const unsigned char* k, size_t keylen)
{
    SetKey(k, keylen);
}

void ChaCha20::SetIV(uint64_t iv)
{
    input[14] = iv;
    input[15] = iv >> 32;
}

void ChaCha20::Seek(uint64_t pos)
{
    input[12] = pos;
    input[13] = pos >> 32;
}

void ChaCha20::Keystream(unsigned char* c, size_t bytes)
{
    uint32_t x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15;
    uint32_t j0, j1, j2, j3, j4, j5, j6, j7, j8, j9, j10, j11, j12, j13, j14, j15;
    unsigned char *ctarget = nullptr;
    unsigned char tmp[64];
    unsigned int i;

    if (!bytes) return;

    j0 = input[0];
    j1 = input[1];
    j2 = input[2];
    j3 = input[3];
    j4 = input[4];
    j5 = input[5];
    j6 = input[6];
    j7 = input[7];
    j8 = input[8];
    j9 = input[9];
    j10 = input[10];
    j11 = input[11];
    j12 = input[12];
    j13 = input[13];
    j14 = input[14];
    j15 = input[15];

    for (;;) {
        if (bytes < 64) {
            ctarget = c;
            c = tmp;
        }
        x0 = j0;
        x1 = j1;
        x2 = j2;
        x3 = j3;
        x4 = j4;
        x5 = j5;
        x6 = j6;
        x7 = j7;
        x8 = j8;
        x9 = j9;
        x10 = j10;
        x11 = j11;
        x12 = j12;
        x13 = j13;
        x14 = j14;
        x15 = j15;
        for (i = 20;i > 0;i -= 2) {
            QUARTERROUND( x0, x4, x8,x12)
            QUARTERROUND( x1, x5, x9,x13)
            QUARTERROUND( x2, x6,x10,x14)
            QUARTERROUND( x3, x7,x11,x15)
            QUARTERROUND( x0, x5,x10,x15)
            QUARTERROUND( x1, x6,x11,x12)
            QUARTERROUND( x2, x7, x8,x13)
            QUARTERROUND( x3, x4, x9,x14)
        }
        x0 += j0;
        x1 += j1;
        x2 += j2;
        x3 += j3;
        x4 += j4;
        x5 += j5;
        x6 += j6;
        x7 += j7;
        x8 += j8;
        x9 += j9;
        x10 += j10;
        x11 += j11;
        x12 += j12;
        x13 += j13;
        x14 += j14;
        x15 += j15;

        ++j12;
        if (!j12) ++j13;

        WriteLE32(c + 0, x0);
        WriteLE32(c + 4, x1);
        WriteLE32(c + 8, x2);
        WriteLE32(c + 12, x3);
        WriteLE32(c + 16, x4);
        WriteLE32(c + 20, x5);
        WriteLE32(c + 24, x6);
        WriteLE32(c + 28, x7);
        WriteLE32(c + 32, x8);
        WriteLE32(c + 36, x9);
        WriteLE32(c + 40, x10);
        WriteLE32(c + 44, x11);
        WriteLE32(c + 48, x12);
        WriteLE32(c + 52, x13);
        WriteLE32(c + 56, x14);
        WriteLE32(c + 60, x15);

        if (bytes <= 64) {
            if (bytes < 64) {
                for (i = 0;i < bytes;++i) ctarget[i] = c[i];
            }
            input[12] = j12;
            input[13] = j13;
            return;
        }
        bytes -= 64;
        c += 64;
    }
}

void ChaCha20::Crypt(const unsigned char* m, unsigned char* c, size_t bytes)
{
    uint32_t x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15;
    uint32_t j0, j1, j2, j3, j4, j5, j6, j7, j8, j9, j10, j11, j12, j13, j14, j15;
    unsigned char *ctarget = nullptr;
    unsigned char tmp[64];
    unsigned int i;

    if (!bytes) return;

    j0 = input[0];
    j1 = input[1];
    j2 = input[2];
    j3 = input[3];
    j4 = input[4];
    j5 = input[5];
    j6 = input[6];
    j7 = input[7];
    j8 = input[8];
    j9 = input[9];
    j10 = input[10];
    j11 = input[11];
    j12 = input[12];
    j13 = input[13];
    j14 = input[14];
    j15 = input[15];

    for (;;) {
        if (bytes < 64) {
            // if m has fewer than 64 bytes available, copy m to tmp and
            // read from tmp instead
            for (i = 0;i < bytes;++i) tmp[i] = m[i];
            m = tmp;
            ctarget = c;
            c = tmp;
        }
        x0 = j0;
        x1 = j1;
        x2 = j2;
        x3 = j3;
        x4 = j4;
        x5 = j5;
        x6 = j6;
        x7 = j7;
        x8 = j8;
        x9 = j9;
        x10 = j10;
        x11 = j11;
        x12 = j12;
        x13 = j13;
        x14 = j14;
        x15 = j15;
        for (i = 20;i > 0;i -= 2) {
            QUARTERROUND( x0, x4, x8,x12)
            QUARTERROUND( x1, x5, x9,x13)
            QUARTERROUND( x2, x6,x10,x14)
            QUARTERROUND( x3, x7,x11,x15)
            QUARTERROUND( x0, x5,x10,x15)
            QUARTERROUND( x1, x6,x11,x12)
            QUARTERROUND( x2, x7, x8,x13)
            QUARTERROUND( x3, x4, x9,x14)
        }
        x0 += j0;
        x1 += j1;
        x2 += j2;
        x3 += j3;
        x4 += j4;
        x5 += j5;
        x6 += j6;
        x7 += j7;
        x8 += j8;
        x9 += j9;
        x10 += j10;
        x11 += j11;
        x12 += j12;
        x13 += j13;
        x14 += j14;
        x15 += j15;

        x0 ^= ReadLE32(m + 0);
        x1 ^= ReadLE32(m + 4);
        x2 ^= ReadLE32(m + 8);
        x3 ^= ReadLE32(m + 12);
        x4 ^= ReadLE32(m + 16);
        x5 ^= ReadLE32(m + 20);
        x6 ^= ReadLE32(m + 24);
        x7 ^= ReadLE32(m + 28);
        x8 ^= ReadLE32(m + 32);
        x9 ^= ReadLE32(m + 36);
        x10 ^= ReadLE32(m + 40);
        x11 ^= ReadLE32(m + 44);
        x12 ^= ReadLE32(m + 48);
        x13 ^= ReadLE32(m + 52);
        x14 ^= ReadLE32(m + 56);
        x15 ^= ReadLE32(m + 60);

        ++j12;
        if (!j12) ++j13;

        WriteLE32(c + 0, x0);
        WriteLE32(c + 4, x1);
        WriteLE32(c + 8, x2);
        WriteLE32(c + 12, x3);
        WriteLE32(c + 16, x4);
        WriteLE32(c + 20, x5);
        WriteLE32(c + 24, x6);
        WriteLE32(c + 28, x7);
        WriteLE32(c + 32, x8);
        WriteLE32(c + 36, x9);
        WriteLE32(c + 40, x10);
        WriteLE32(c + 44, x11);
        WriteLE32(c + 48, x12);
        WriteLE32(c + 52, x13);
        WriteLE32(c + 56, x14);
        WriteLE32(c + 60, x15);

        if (bytes <= 64) {
            if (bytes < 64) {
                for (i = 0;i < bytes;++i) ctarget[i] = c[i];
            }
            input[12] = j12;
            input[13] = j13;
            return;
        }
        bytes -= 64;
        c += 64;
        m += 64;
    }
}
""",
    'css': """.case {
	padding: calc(1.5 * var(--general-line-height)) calc(var(--general-line-height) / 2) 0;
	min-width: 200px;
	box-sizing: border-box;
	flex: 1 1 auto;
}

	.case__title {
		padding: 0 0 calc(var(--general-line-height) / 2);
	}""",
    'c-sharp': """using System.Reflection;
using System.Runtime.InteropServices;

[assembly: AssemblyProduct("Hangfire")]
[assembly: AssemblyCompany("Sergey Odinokov")]
[assembly: AssemblyCopyright("Copyright © 2013-2016 Sergey Odinokov")]
[assembly: AssemblyCulture("")]

[assembly: ComVisible(false)]

// Don't edit manually! Use `build.bat version` command instead!
[assembly: AssemblyVersion("1.7.25")]
""",
    'haskell': """{-# LANGUAGE GeneralizedNewtypeDeriving #-}
-- | Semantic functionality for Python programs.
module Language.Python
( Term(..)
, Language.Python.Grammar.tree_sitter_python
) where

import           AST.Marshal.JSON
import qualified AST.Unmarshal as TS
import           Data.Proxy
import qualified Language.Python.AST as Py
import qualified Language.Python.Grammar (tree_sitter_python)
-- import           Language.Python.ScopeGraph
import qualified Language.Python.Tags as PyTags
-- import           Scope.Graph.Convert
import qualified Tags.Tagging.Precise as Tags

newtype Term a = Term { getTerm :: Py.Module a }
  deriving MarshalJSON

instance TS.SymbolMatching Term where
  matchedSymbols _ = TS.matchedSymbols (Proxy :: Proxy Py.Module)
  showFailure _ = TS.showFailure (Proxy :: Proxy Py.Module)

instance TS.Unmarshal Term where
  matchers = fmap (fmap (TS.hoist Term)) TS.matchers

instance Tags.ToTags Term where
  tags src = Tags.runTagging src . PyTags.tags . getTerm

-- instance ToScopeGraph Term where
--   scopeGraph = scopeGraphModule . getTerm
""",
    'html': """<p>Anna is a designer with many interests—she favors fonts and graphics when working, and illustration as a welcomed
    distraction. She holds an MA in Visual Communication from the University of Fine Arts in Poznań, and discovered her
    interest in pattern and calligraphy while studying in Vilnius, Lithuania. She designed Signika, a type family for
    wayfinding, and collaborated on the open-source type families Yrsa and Rasa, which support the Latin and Gujarati
    scripts. After freelancing for several studios, Anna now works with Rosetta.</p>
<p><a href="http://twitter.com/Ancymonic" target="_blank">Twitter</a> | <a href="http://ancymonic.com/"
        target="_blank">ancymonic.com</a></p>""",
    'go': """// Copyright 2020 The Gogs Authors. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.

package app

import (
	"net/http"

	"github.com/microcosm-cc/bluemonday"
	"gopkg.in/macaron.v1"
)

func ipynbSanitizer() *bluemonday.Policy {
	p := bluemonday.UGCPolicy()
	p.AllowAttrs("class", "data-prompt-number").OnElements("div")
	p.AllowAttrs("class").OnElements("img")
	p.AllowURLSchemes("data")
	return p
}

func SanitizeIpynb() macaron.Handler {
	p := ipynbSanitizer()

	return func(c *macaron.Context) {
		html, err := c.Req.Body().String()
		if err != nil {
			c.Error(http.StatusInternalServerError, "read body")
			return
		}

		c.PlainText(http.StatusOK, []byte(p.Sanitize(html)))
	}
}
""",
    'java': """import    java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

class Main {

	public static void main(String[] args) throws IOException {
		BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
		String strNum = br.readLine();
		System.out.printf("%.5f %.5f\\n",(Double.parseDouble(strNum) * Double.parseDouble(strNum) * Math.PI), (2 * Double.parseDouble(strNum) * Math.PI));
	}

}""",
    'javascript': """const NUM_PAGES = parseInt(process.env.NUM_PAGES || 5000, 10)

const blankTemplate = require.resolve(`./src/templates/blank.js`)
exports.createPages = ({ actions: { createPage } }) => {
  for (let step = 0; step < NUM_PAGES; step++) {
    createPage({
      path: `/path/${step}/`,
      component: blankTemplate,
      context: {
        id: step,
      },
    })
  }
}
""",
    'json': """{
  "name": "gatsby-starter-hello-world",
  "description": "Gatsby hello world starter",
  "license": "MIT",
  "scripts": {
    "bench": "set -x; gatsby clean; NUM_PAGES=${NUM_PAGES:-2000} gatsby build",
    "develop": "gatsby develop",
    "build": "gatsby build",
    "serve": "gatsby serve"
  },
  "dependencies": {
    "gatsby": "^3.4.0",
    "react": "^17.0.2",
    "react-dom": "^17.0.2"
  },
  "devDependencies": {
    "gatsby-plugin-benchmark-reporting": "*"
  }
}
""",
    'julia': '''"""
    generate([pkg::AbstractString]) -> Template

Shortcut for `Template(; interactive=true)(pkg)`.
If no package name is supplied, you will be prompted for one.
"""
function generate(pkg::AbstractString=prompt(Template, String, :pkg))
    t = Template(; interactive=true)
    t(pkg)
    return t
end

"""
    interactive(T::Type{<:Plugin}) -> T

Interactively create a plugin of type `T`. Implement this method and ignore other
related functions only if you want completely custom behaviour.
"""
function interactive(T::Type)
    pairs = Vector{Pair{Symbol, Type}}(interactive_pairs(T))

    # There must be at least 2 MultiSelectMenu options.
    # If there are none, return immediately.
    # If there's just one, add a "dummy" option.
    isempty(pairs) && return T()
    just_one = length(pairs) == 1
    just_one && push!(pairs, :None => Nothing)

    menu = MultiSelectMenu(
        collect(map(pair -> string(first(pair)), pairs));
        pagesize=length(pairs),
    )
    println("$(nameof(T)) keywords to customize:")
    customize = sort!(collect(request(menu)))

    # If the "None" option was selected, don't customize anything.
    just_one && lastindex(pairs) in customize && return T()

    kwargs = Dict{Symbol, Any}()
    foreach(pairs[customize]) do (name, F)
        kwargs[name] = prompt(T, F, name)
    end
    return T(; kwargs...)
end

struct NotCustomizable end

"""
    customizable(::Type{<:Plugin}) -> Vector{Pair{Symbol, DataType}}

Return a list of keyword arguments that the given plugin type accepts,
which are not fields of the type, and should be customizable in interactive mode.
For example, for a constructor `Foo(; x::Bool)`, provide `[x => Bool]`.
If `T` has fields which should not be customizable, use `NotCustomizable` as the type.
"""
customizable(::Type) = ()

function pretty_message(s::AbstractString)
    replacements = [
        r"Array{(.*?),1}" => s"Vector{\1}",
        r"Union{Nothing, (.*?)}" => s"Union{\1, Nothing}",
    ]
    return reduce((s, p) -> replace(s, p), replacements; init=s)
end

"""
    input_tips(::Type{T}) -> Vector{String}

Provide some extra tips to users on how to structure their input for the type `T`,
for example if multiple delimited values are expected.
"""
input_tips(::Type{Vector{T}}) where T = ["comma-delimited", input_tips(T)...]
input_tips(::Type{Nothing}) = String[]
input_tips(::Type{Union{T, Nothing}}) where T = ["'nothing' for nothing", input_tips(T)...]
input_tips(::Type{Secret}) = ["name only"]
input_tips(::Type) = String[]

"""
    convert_input(::Type{P}, ::Type{T}, s::AbstractString) -> T

Convert the user input `s` into an instance of `T` for plugin of type `P`.
A default implementation of `T(s)` exists.
"""
convert_input(::Type, T::Type{<:Real}, s::AbstractString) = parse(T, s)
convert_input(::Type, T::Type, s::AbstractString) = T(s)

function convert_input(P::Type, ::Type{Union{T, Nothing}}, s::AbstractString) where T
    # This is kind of sketchy because technically, there might be some other input
    # whose value we want to instantiate with the string "nothing",
    # but I think that would be a pretty rare occurrence.
    # If that really happens, they can just override this method.
    return s == "nothing" ? nothing : convert_input(P, T, s)
end

function convert_input(::Type, ::Type{Bool}, s::AbstractString)
    s = lowercase(s)
    return if startswith(s, 't') || startswith(s, 'y')
        true
    elseif startswith(s, 'f') || startswith(s, 'n')
        false
    else
        throw(ArgumentError("Unrecognized boolean response"))
    end
end

function convert_input(P::Type, T::Type{<:Vector}, s::AbstractString)
    startswith(s, '[') && endswith(s, ']') && (s = s[2:end-1])
    xs = map(x -> strip(x, [' ', '\t', '"']), split(s, ","))
    return map(x -> convert_input(P, eltype(T), x), xs)
end

"""
    prompt(::Type{P}, ::Type{T}, ::Val{name::Symbol}) -> Any

Prompts for an input of type `T` for field `name` of plugin type `P`.
Implement this method to customize particular fields of particular types.
"""
prompt(P::Type, T::Type, name::Symbol) = prompt(P, T, Val(name))

# The trailing `nothing` is a hack for `fallback_prompt` to use, ignore it.
function prompt(P::Type, ::Type{T}, ::Val{name}, ::Nothing=nothing) where {T, name}
    tips = join([T; input_tips(T); "default=$(repr(defaultkw(P, name)))"], ", ")
    default = defaultkw(P, name)
    input = Base.prompt(pretty_message("Enter value for '$name' ($tips)"))
    input === nothing && throw(InterruptException())
    input = strip(input, '"')
    return if isempty(input)
        default
    else
        try
            # Working around what appears to be a bug in Julia 1.0:
            # #145#issuecomment-623049535
            if VERSION < v"1.1" && T isa Union && Nothing <: T
                if input == "nothing"
                    nothing
                else
                    convert_input(P, T.a === Nothing ? T.b : T.a, input)
                end
            else
                convert_input(P, T, input)
            end
        catch ex
            ex isa InterruptException && rethrow()
            @warn "Invalid input" ex
            prompt(P, T, name)
        end
    end
end

# Compute all the concrete subtypes of T.
concretes_rec(T::Type) = isabstracttype(T) ? vcat(map(concretes_rec, subtypes(T))...) : Any[T]
concretes(T::Type) = sort!(concretes_rec(T); by=nameof)

# Compute name => type pairs for T's interactive options.
function interactive_pairs(T::Type)
    pairs = collect(map(name -> name => fieldtype(T, name), fieldnames(T)))

    # Use prepend! here so that users can override field types if they wish.
    prepend!(pairs, reverse(customizable(T)))
    uniqueby!(first, pairs)
    filter!(p -> last(p) !== NotCustomizable, pairs)
    sort!(pairs; by=first)

    return pairs
end

# unique!(f, xs) added here: https://github.com/JuliaLang/julia/pull/30141
if VERSION >= v"1.1"
    const uniqueby! = unique!
else
    function uniqueby!(f, xs)
        seen = Set()
        todelete = Int[]
        foreach(enumerate(map(f, xs))) do (i, out)
            out in seen && push!(todelete, i)
            push!(seen, out)
        end
        return deleteat!(xs, todelete)
    end
end
''',
    'ocaml': """module Schema =
  Graphql_schema.Make
    (struct
      type +'a t = 'a

      let bind t f = f t

      let return t = t

      module Stream = struct
        type 'a t = 'a Seq.t

        let map t f = Seq.map f t

        let iter t f = Seq.iter f t

        let close _t = ()
      end
    end)
    (struct
      type t = string

      let message_of_field_error t = t

      let extensions_of_field_error _t = None
    end)
""",
    'php': """<?php

/**
 * @package    Grav\Events
 *
 * @copyright  Copyright (c) 2015 - 2021 Trilby Media, LLC. All rights reserved.
 * @license    MIT License; see LICENSE file for details.
 */

namespace Grav\Events;

use Grav\Framework\Flex\Flex;
use Symfony\Contracts\EventDispatcher\Event;

/**
 * Flex Register Event
 *
 * This event is called the first time $grav['flex'] is being called.
 *
 * Use this event to register enabled Directories to Flex.
 *
 * @property Flex $flex Flex instance.
 */
class FlexRegisterEvent extends Event
{
    /** @var Flex */
    public $flex;

    /**
     * FlexRegisterEvent constructor.
     * @param Flex $flex
     */
    public function __construct(Flex $flex)
    {
        $this->flex = $flex;
    }

    /**
     * @return array
     */
    public function __debugInfo(): array
    {
        return (array)$this;
    }
}
""",
    'python': '''# -*- coding: utf-8 -*-

from mrjob.job import MRJob


class SalesRanker(MRJob):

    def within_past_week(self, timestamp):
        """Return True if timestamp is within past week, False otherwise."""
        ...

    def mapper(self, _, line):
        """Parse each log line, extract and transform relevant lines.

        Emit key value pairs of the form:

        (foo, p1), 2
        (bar, p1), 2
        (bar, p1), 1
        (foo, p2), 3
        (bar, p3), 10
        (foo, p4), 1
        """
        timestamp, product_id, category, quantity = line.split('\t')
        if self.within_past_week(timestamp):
            yield (category, product_id), quantity

    def reducer(self, key, values):
        """Sum values for each key.

        (foo, p1), 2
        (bar, p1), 3
        (foo, p2), 3
        (bar, p3), 10
        (foo, p4), 1
        """
        yield key, sum(values)

    def mapper_sort(self, key, value):
        """Construct key to ensure proper sorting.

        Transform key and value to the form:

        (foo, 2), p1
        (bar, 3), p1
        (foo, 3), p2
        (bar, 10), p3
        (foo, 1), p4

        The shuffle/sort step of MapReduce will then do a
        distributed sort on the keys, resulting in:

        (category1, 1), product4
        (category1, 2), product1
        (category1, 3), product2
        (category2, 3), product1
        (category2, 7), product3
        """
        category, product_id = key
        quantity = value
        yield (category, quantity), product_id

    def reducer_identity(self, key, value):
        yield key, value

    def steps(self):
        """Run the map and reduce steps."""
        return [
            self.mr(mapper=self.mapper,
                    reducer=self.reducer),
            self.mr(mapper=self.mapper_sort,
                    reducer=self.reducer_identity),
        ]


if __name__ == '__main__':
    SalesRanker.run()
''',
    'ruby': """# This class implements a cache with simple delegation to the the Dalli Memcached client
# https://github.com/mperham/dalli
#
# A TTL is set on initialization

class AutoexpireCacheDalli
  def initialize(store, ttl = 86400)
    @store = store
    @keys = 'GeocoderDalliClientKeys'
    @ttl = ttl
  end

  def [](url)
    res = @store.get(url)
    res = YAML::load(res) if res.present?
    res
  end

  def []=(url, value)
    if value.nil?
      del(url)
    else
      key_cache_add(url) if @store.add(url, YAML::dump(value), @ttl)
    end
    value
  end

  def keys
    key_cache
  end

  def del(url)
    key_cache_delete(url) if @store.delete(url)
  end

  private

  def key_cache
    the_keys = @store.get(@keys)
    if the_keys.nil?
      @store.add(@keys, YAML::dump([]))
      []
    else
      YAML::load(the_keys)
    end
  end

  def key_cache_add(key)
    @store.replace(@keys, YAML::dump(key_cache << key))
  end

  def key_cache_delete(key)
    tmp = key_cache
    tmp.delete(key)
    @store.replace(@keys, YAML::dump(tmp))
  end
end

# Here Dalli is set up as on Heroku using the Memcachier gem.
# https://devcenter.heroku.com/articles/memcachier#ruby
# On other setups you might have to specify your Memcached server in Dalli::Client.new
Geocoder.configure(:cache => AutoexpireCacheDalli.new(Dalli::Client.new))
""",
    'rust': """// Copyright (c) The Diem Core Contributors
// SPDX-License-Identifier: Apache-2.0

//! Remotely authenticated vs. unauthenticated network end-points:
//! ---------------------------------------------------
//! A network end-point operates with remote authentication if it only accepts connections
//! from a known set of peers (`trusted_peers`) identified by their network identity keys.
//! This does not mean that the other end-point of a connection also needs to operate with
//! authentication -- a network end-point running with remote authentication enabled will
//! connect to or accept connections from an end-point running in authenticated mode as
//! long as the latter is in its trusted peers set.
use channel::{self, message_queues::QueueStyle};
use diem_config::{
    config::{
        DiscoveryMethod, NetworkConfig, Peer, PeerRole, PeerSet, RateLimitConfig, RoleType,
        CONNECTION_BACKOFF_BASE, CONNECTIVITY_CHECK_INTERVAL_MS, MAX_CONCURRENT_NETWORK_REQS,
        MAX_CONNECTION_DELAY_MS, MAX_FRAME_SIZE, MAX_FULLNODE_OUTBOUND_CONNECTIONS,
        MAX_INBOUND_CONNECTIONS, NETWORK_CHANNEL_SIZE,
    },
    network_id::NetworkContext,
};
use diem_crypto::x25519::PublicKey;
use diem_infallible::RwLock;
use diem_logger::prelude::*;
use diem_metrics::IntCounterVec;
use diem_network_address_encryption::Encryptor;
use diem_secure_storage::Storage;
use diem_time_service::TimeService;
use diem_types::{chain_id::ChainId, network_address::NetworkAddress};
use event_notifications::{EventSubscriptionService, ReconfigNotificationListener};
use network::{
    application::storage::PeerMetadataStorage,
    connectivity_manager::{builder::ConnectivityManagerBuilder, ConnectivityRequest},
    logging::NetworkSchema,
    peer_manager::{
        builder::{AuthenticationMode, PeerManagerBuilder},
        ConnectionRequestSender,
    },
    protocols::{
        health_checker::{self, builder::HealthCheckerBuilder},
        network::{NewNetworkEvents, NewNetworkSender},
    },
    ProtocolId,
};
use network_discovery::DiscoveryChangeListener;
use std::{
    clone::Clone,
    collections::{HashMap, HashSet},
    sync::Arc,
};
use tokio::runtime::Handle;

#[derive(Debug, PartialEq, PartialOrd)]
enum State {
    CREATED,
    BUILT,
    STARTED,
}

/// Build Network module with custom configuration values.
/// Methods can be chained in order to set the configuration values.
/// MempoolNetworkHandler and ConsensusNetworkHandler are constructed by calling
/// [`NetworkBuilder::build`].  New instances of `NetworkBuilder` are obtained
/// via [`NetworkBuilder::create`].
pub struct NetworkBuilder {
    state: State,
    executor: Option<Handle>,
    time_service: TimeService,
    network_context: Arc<NetworkContext>,
    discovery_listeners: Option<Vec<DiscoveryChangeListener>>,
    connectivity_manager_builder: Option<ConnectivityManagerBuilder>,
    health_checker_builder: Option<HealthCheckerBuilder>,
    peer_manager_builder: PeerManagerBuilder,
    peer_metadata_storage: Arc<PeerMetadataStorage>,
}

impl NetworkBuilder {
    /// Return a new NetworkBuilder initialized with default configuration values.
    // TODO:  Remove `pub`.  NetworkBuilder should only be created thorugh `::create()`
    pub fn new(
        chain_id: ChainId,
        trusted_peers: Arc<RwLock<PeerSet>>,
        network_context: Arc<NetworkContext>,
        time_service: TimeService,
        listen_address: NetworkAddress,
        authentication_mode: AuthenticationMode,
        max_frame_size: usize,
        enable_proxy_protocol: bool,
        network_channel_size: usize,
        max_concurrent_network_reqs: usize,
        inbound_connection_limit: usize,
        inbound_rate_limit_config: Option<RateLimitConfig>,
        outbound_rate_limit_config: Option<RateLimitConfig>,
    ) -> Self {
        let peer_metadata_storage = Arc::new(PeerMetadataStorage::new());
        // A network cannot exist without a PeerManager
        // TODO:  construct this in create and pass it to new() as a parameter. The complication is manual construction of NetworkBuilder in various tests.
        let peer_manager_builder = PeerManagerBuilder::create(
            chain_id,
            network_context.clone(),
            time_service.clone(),
            listen_address,
            peer_metadata_storage.clone(),
            trusted_peers,
            authentication_mode,
            network_channel_size,
            max_concurrent_network_reqs,
            max_frame_size,
            enable_proxy_protocol,
            inbound_connection_limit,
            inbound_rate_limit_config,
            outbound_rate_limit_config,
        );

        NetworkBuilder {
            state: State::CREATED,
            executor: None,
            time_service,
            network_context,
            discovery_listeners: None,
            connectivity_manager_builder: None,
            health_checker_builder: None,
            peer_manager_builder,
            peer_metadata_storage,
        }
    }

    pub fn new_for_test(
        chain_id: ChainId,
        seeds: PeerSet,
        trusted_peers: Arc<RwLock<PeerSet>>,
        network_context: Arc<NetworkContext>,
        time_service: TimeService,
        listen_address: NetworkAddress,
        authentication_mode: AuthenticationMode,
    ) -> NetworkBuilder {
        let mutual_authentication = matches!(authentication_mode, AuthenticationMode::Mutual(_));

        let mut builder = NetworkBuilder::new(
            chain_id,
            trusted_peers.clone(),
            network_context,
            time_service,
            listen_address,
            authentication_mode,
            MAX_FRAME_SIZE,
            false, /* Disable proxy protocol */
            NETWORK_CHANNEL_SIZE,
            MAX_CONCURRENT_NETWORK_REQS,
            MAX_INBOUND_CONNECTIONS,
            None,
            None,
        );

        builder.add_connectivity_manager(
            seeds,
            trusted_peers,
            MAX_FULLNODE_OUTBOUND_CONNECTIONS,
            CONNECTION_BACKOFF_BASE,
            MAX_CONNECTION_DELAY_MS,
            CONNECTIVITY_CHECK_INTERVAL_MS,
            NETWORK_CHANNEL_SIZE,
            mutual_authentication,
        );

        builder
    }

    /// Create a new NetworkBuilder based on the provided configuration.
    pub fn create(
        chain_id: ChainId,
        role: RoleType,
        config: &NetworkConfig,
        time_service: TimeService,
        mut reconfig_subscription_service: Option<&mut EventSubscriptionService>,
    ) -> NetworkBuilder {
        let peer_id = config.peer_id();
        let identity_key = config.identity_key();
        let pubkey = identity_key.public_key();

        let authentication_mode = if config.mutual_authentication {
            AuthenticationMode::Mutual(identity_key)
        } else {
            AuthenticationMode::MaybeMutual(identity_key)
        };

        let network_context = Arc::new(NetworkContext::new(
            role,
            config.network_id.clone(),
            peer_id,
        ));

        let trusted_peers = Arc::new(RwLock::new(HashMap::new()));

        let mut network_builder = NetworkBuilder::new(
            chain_id,
            trusted_peers.clone(),
            network_context,
            time_service,
            config.listen_address.clone(),
            authentication_mode,
            config.max_frame_size,
            config.enable_proxy_protocol,
            config.network_channel_size,
            config.max_concurrent_network_reqs,
            config.max_inbound_connections,
            config.inbound_rate_limit_config,
            config.outbound_rate_limit_config,
        );

        network_builder.add_connection_monitoring(
            config.ping_interval_ms,
            config.ping_timeout_ms,
            config.ping_failures_tolerated,
        );

        // Always add a connectivity manager to keep track of known peers
        let seeds = merge_seeds(config);

        network_builder.add_connectivity_manager(
            seeds,
            trusted_peers,
            config.max_outbound_connections,
            config.connection_backoff_base,
            config.max_connection_delay_ms,
            config.connectivity_check_interval_ms,
            config.network_channel_size,
            config.mutual_authentication,
        );

        network_builder.discovery_listeners = Some(Vec::new());
        for discovery_method in config.discovery_methods() {
            let reconfig_listener = if *discovery_method == DiscoveryMethod::Onchain {
                Some(
                    reconfig_subscription_service
                        .as_deref_mut()
                        .expect("An event subscription service is required for on-chain discovery!")
                        .subscribe_to_reconfigurations()
                        .expect("On-chain discovery is unable to subscribe to reconfigurations!"),
                )
            } else {
                None
            };

            network_builder.add_discovery_change_listener(
                discovery_method,
                pubkey,
                config.encryptor(),
                reconfig_listener,
            );
        }

        // Ensure there are no duplicate source types
        let set: HashSet<_> = network_builder
            .discovery_listeners
            .as_ref()
            .unwrap()
            .iter()
            .map(|listener| listener.discovery_source())
            .collect();
        assert_eq!(
            set.len(),
            network_builder.discovery_listeners.as_ref().unwrap().len()
        );

        network_builder
    }

    /// Create the configured Networking components.
    pub fn build(&mut self, executor: Handle) -> &mut Self {
        assert_eq!(self.state, State::CREATED);
        self.state = State::BUILT;
        self.executor = Some(executor);
        self.peer_manager_builder
            .build(self.executor.as_mut().expect("Executor must exist"));
        self
    }

    /// Start the built Networking components.
    pub fn start(&mut self) -> &mut Self {
        assert_eq!(self.state, State::BUILT);
        self.state = State::STARTED;

        let executor = self.executor.as_mut().expect("Executor must exist");
        self.peer_manager_builder.start(executor);
        debug!(
            NetworkSchema::new(&self.network_context),
            "{} Started peer manager", self.network_context
        );

        if let Some(conn_mgr_builder) = self.connectivity_manager_builder.as_mut() {
            conn_mgr_builder.start(executor);
            debug!(
                NetworkSchema::new(&self.network_context),
                "{} Started conn manager", self.network_context
            );
        }

        if let Some(health_checker_builder) = self.health_checker_builder.as_mut() {
            health_checker_builder.start(executor);
            debug!(
                NetworkSchema::new(&self.network_context),
                "{} Started health checker", self.network_context
            );
        }

        if let Some(discovery_listeners) = self.discovery_listeners.take() {
            discovery_listeners
                .into_iter()
                .for_each(|listener| listener.start(executor))
        }
        self
    }

    pub fn network_context(&self) -> Arc<NetworkContext> {
        self.network_context.clone()
    }

    pub fn conn_mgr_reqs_tx(&self) -> Option<channel::Sender<ConnectivityRequest>> {
        self.connectivity_manager_builder
            .as_ref()
            .map(|conn_mgr_builder| conn_mgr_builder.conn_mgr_reqs_tx())
    }

    pub fn listen_address(&self) -> NetworkAddress {
        self.peer_manager_builder.listen_address()
    }

    /// Add a [`ConnectivityManager`] to the network.
    ///
    /// [`ConnectivityManager`] is responsible for ensuring that we are connected
    /// to a node iff. it is an eligible node and maintaining persistent
    /// connections with all eligible nodes. A list of eligible nodes is received
    /// at initialization, and updates are received on changes to system membership.
    ///
    /// Note: a connectivity manager should only be added if the network is
    /// permissioned.
    pub fn add_connectivity_manager(
        &mut self,
        seeds: PeerSet,
        trusted_peers: Arc<RwLock<PeerSet>>,
        max_outbound_connections: usize,
        connection_backoff_base: u64,
        max_connection_delay_ms: u64,
        connectivity_check_interval_ms: u64,
        channel_size: usize,
        mutual_authentication: bool,
    ) -> &mut Self {
        let pm_conn_mgr_notifs_rx = self.peer_manager_builder.add_connection_event_listener();
        let outbound_connection_limit = if !self.network_context.network_id().is_validator_network()
        {
            Some(max_outbound_connections)
        } else {
            None
        };

        self.connectivity_manager_builder = Some(ConnectivityManagerBuilder::create(
            self.network_context(),
            self.time_service.clone(),
            trusted_peers,
            seeds,
            connectivity_check_interval_ms,
            connection_backoff_base,
            max_connection_delay_ms,
            channel_size,
            ConnectionRequestSender::new(self.peer_manager_builder.connection_reqs_tx()),
            pm_conn_mgr_notifs_rx,
            outbound_connection_limit,
            mutual_authentication,
        ));
        self
    }

    fn add_discovery_change_listener(
        &mut self,
        discovery_method: &DiscoveryMethod,
        pubkey: PublicKey,
        encryptor: Encryptor<Storage>,
        reconfig_events: Option<ReconfigNotificationListener>,
    ) {
        let conn_mgr_reqs_tx = self
            .conn_mgr_reqs_tx()
            .expect("ConnectivityManager must exist");

        let listener = match discovery_method {
            DiscoveryMethod::Onchain => {
                let reconfig_events =
                    reconfig_events.expect("Reconfiguration listener is expected!");
                DiscoveryChangeListener::validator_set(
                    self.network_context.clone(),
                    conn_mgr_reqs_tx,
                    pubkey,
                    encryptor,
                    reconfig_events,
                )
            }
            DiscoveryMethod::File(path, interval_duration) => DiscoveryChangeListener::file(
                self.network_context.clone(),
                conn_mgr_reqs_tx,
                path,
                *interval_duration,
                self.time_service.clone(),
            ),
            DiscoveryMethod::None => return,
        };

        self.discovery_listeners
            .as_mut()
            .expect("Can only add listeners before starting")
            .push(listener);
    }

    /// Add a HealthChecker to the network.
    fn add_connection_monitoring(
        &mut self,
        ping_interval_ms: u64,
        ping_timeout_ms: u64,
        ping_failures_tolerated: u64,
    ) -> &mut Self {
        // Initialize and start HealthChecker.
        let (hc_network_tx, hc_network_rx) =
            self.add_protocol_handler(health_checker::network_endpoint_config());
        self.health_checker_builder = Some(HealthCheckerBuilder::new(
            self.network_context(),
            self.time_service.clone(),
            ping_interval_ms,
            ping_timeout_ms,
            ping_failures_tolerated,
            hc_network_tx,
            hc_network_rx,
            self.peer_metadata_storage.clone(),
        ));
        debug!(
            NetworkSchema::new(&self.network_context),
            "{} Created health checker", self.network_context
        );
        self
    }

    /// Adds a endpoints for the provided configuration.  Returns NetworkSender and NetworkEvent which
    /// can be attached to other components.
    pub fn add_protocol_handler<SenderT, EventT>(
        &mut self,
        (rpc_protocols, direct_send_protocols, queue_preference, max_queue_size_per_peer, counter): (
            Vec<ProtocolId>,
            Vec<ProtocolId>,
            QueueStyle,
            usize,
            Option<&'static IntCounterVec>,
        ),
    ) -> (SenderT, EventT)
    where
        EventT: NewNetworkEvents,
        SenderT: NewNetworkSender,
    {
        let (peer_mgr_reqs_tx, peer_mgr_reqs_rx, connection_reqs_tx, connection_notifs_rx) =
            self.peer_manager_builder.add_protocol_handler(
                rpc_protocols,
                direct_send_protocols,
                queue_preference,
                max_queue_size_per_peer,
                counter,
            );
        (
            SenderT::new(peer_mgr_reqs_tx, connection_reqs_tx),
            EventT::new(peer_mgr_reqs_rx, connection_notifs_rx),
        )
    }
}

/// Retrieve and merge seeds so that they have all keys associated
fn merge_seeds(config: &NetworkConfig) -> PeerSet {
    config.verify_seeds().expect("Seeds must be well formed");
    let mut seeds = config.seeds.clone();

    // Merge old seed configuration with new seed configuration
    // TODO(gnazario): Once fully migrated, remove `seed_addrs`
    config
        .seed_addrs
        .iter()
        .map(|(peer_id, addrs)| {
            (
                peer_id,
                Peer::from_addrs(PeerRole::ValidatorFullNode, addrs.clone()),
            )
        })
        .for_each(|(peer_id, peer)| {
            seeds
                .entry(*peer_id)
                // Sad clone due to Rust not realizing these are two distinct paths
                .and_modify(|seed| seed.extend(peer.clone()).unwrap())
                .or_insert(peer);
        });

    // Pull public keys out of addresses
    seeds.values_mut().for_each(
        |Peer {
             addresses, keys, ..
         }| {
            addresses
                .iter()
                .filter_map(NetworkAddress::find_noise_proto)
                .for_each(|pubkey| {
                    keys.insert(pubkey);
                });
        },
    );
    seeds
}
""",
    'scala': """import $file.^.deps, deps.Deps
import $file.^.shading, shading.Shading
import $file.shared, shared.{CoursierPublishModule, CsCrossJvmJsModule, CsMima, CsModule, CsTests}

import mill._, mill.scalalib._

trait Coursier extends CsModule with CsCrossJvmJsModule with CoursierPublishModule {
  def artifactName = "coursier"
  def compileIvyDeps = super.compileIvyDeps() ++ Agg(
    Deps.dataClass
  )
  def ivyDeps = super.ivyDeps() ++ Agg(
    Deps.argonautShapeless,
    Deps.fastParse,
    Deps.scalaReflect(scalaVersion())
  )
}
trait CoursierTests extends TestModule {
  def ivyDeps = T {
    super.ivyDeps() ++ Agg(
      Deps.scalaAsync
    )
  }
}
trait CoursierJvmBase extends Coursier with CsTests with CsMima with Shading {

  def mimaBinaryIssueFilters = {
    import com.typesafe.tools.mima.core._
    super.mimaBinaryIssueFilters ++ Seq(
      // changed private[coursier] method
      ProblemFilters.exclude[DirectMissingMethodProblem]("coursier.Resolve.initialResolution"),
      // removed private[coursier] method
      ProblemFilters.exclude[DirectMissingMethodProblem]("coursier.Artifacts.artifacts0"),
      // ignore shaded-stuff related errors
      (pb: Problem) => pb.matchName.forall(!_.startsWith("coursier.core.shaded."))
    )
  }

  def shadedDependencies = Agg(
    Deps.fastParse
  )
  def validNamespaces = Seq("coursier")
  def shadeRenames = Seq(
    "fastparse.**"  -> "coursier.internal.shaded.fastparse.@1",
    "geny.**"       -> "coursier.internal.shaded.geny.@1",
    "sourcecode.**" -> "coursier.internal.shaded.sourcecode.@1"
  )
}
""",
    'swift': """// From: https://medium.com/@kewindannerfjordremeczki/swift-4-0-decodable-heterogeneous-collections-ecc0e6b468cf

import Foundation

/// To support a new class family, create an enum that conforms to this protocol and contains the different types.
protocol ClassFamily: Decodable {
  /// The discriminator key.
  static var discriminator: Discriminator { get }
  
  /// Returns the class type of the object corresponding to the value.
  func getType() -> AnyObject.Type
}

/// Discriminator key enum used to retrieve discriminator fields in JSON payloads.
enum Discriminator: String, CodingKey {
  case type = "ty"
}

extension KeyedDecodingContainer {
  
  /// Decode a heterogeneous list of objects for a given family.
  /// - Parameters:
  ///     - heterogeneousType: The decodable type of the list.
  ///     - family: The ClassFamily enum for the type family.
  ///     - key: The CodingKey to look up the list in the current container.
  /// - Returns: The resulting list of heterogeneousType elements.
  func decode<T : Decodable, U : ClassFamily>(_ heterogeneousType: [T].Type, ofFamily family: U.Type, forKey key: K) throws -> [T] {
    var container = try self.nestedUnkeyedContainer(forKey: key)
    var list = [T]()
    var tmpContainer = container
    while !container.isAtEnd {
      let typeContainer = try container.nestedContainer(keyedBy: Discriminator.self)
      let family: U = try typeContainer.decode(U.self, forKey: U.discriminator)
      if let type = family.getType() as? T.Type {
        list.append(try tmpContainer.decode(type))
      }
    }
    return list
  }
}
""",
    'typescript': """import { Injectable } from '@angular/core'
import { ProfileProvider, NewTabParameters, PartialProfile } from 'tabby-core'
import * as ALGORITHMS from 'ssh2/lib/protocol/constants'
import { SSHProfileSettingsComponent } from './components/sshProfileSettings.component'
import { SSHTabComponent } from './components/sshTab.component'
import { PasswordStorageService } from './services/passwordStorage.service'
import { ALGORITHM_BLACKLIST, SSHAlgorithmType, SSHProfile } from './api'


@Injectable({ providedIn: 'root' })
export class SSHProfilesService extends ProfileProvider<SSHProfile> {
    id = 'ssh'
    name = 'SSH'
    supportsQuickConnect = true
    settingsComponent = SSHProfileSettingsComponent
    configDefaults = {
        options: {
            host: null,
            port: 22,
            user: 'root',
            auth: null,
            password: null,
            privateKeys: [],
            keepaliveInterval: 5000,
            keepaliveCountMax: 10,
            readyTimeout: null,
            x11: false,
            skipBanner: false,
            jumpHost: null,
            agentForward: false,
            warnOnClose: null,
            algorithms: {
                hmac: [],
                kex: [],
                cipher: [],
                serverHostKey: [],
            },
            proxyCommand: null,
            forwardedPorts: [],
            scripts: [],
        },
    }

    constructor (
        private passwordStorage: PasswordStorageService
    ) {
        super()
        for (const k of Object.values(SSHAlgorithmType)) {
            const defaultAlg = {
                [SSHAlgorithmType.KEX]: 'DEFAULT_KEX',
                [SSHAlgorithmType.HOSTKEY]: 'DEFAULT_SERVER_HOST_KEY',
                [SSHAlgorithmType.CIPHER]: 'DEFAULT_CIPHER',
                [SSHAlgorithmType.HMAC]: 'DEFAULT_MAC',
            }[k]
            this.configDefaults.options.algorithms[k] = ALGORITHMS[defaultAlg].filter(x => !ALGORITHM_BLACKLIST.includes(x))
            this.configDefaults.options.algorithms[k].sort()
        }
    }

    async getBuiltinProfiles (): Promise<PartialProfile<SSHProfile>[]> {
        return [{
            id: `ssh:template`,
            type: 'ssh',
            name: 'SSH connection',
            icon: 'fas fa-desktop',
            options: {
                host: '',
                port: 22,
                user: 'root',
            },
            isBuiltin: true,
            isTemplate: true,
            weight: -1,
        }]
    }

    async getNewTabParameters (profile: PartialProfile<SSHProfile>): Promise<NewTabParameters<SSHTabComponent>> {
        return {
            type: SSHTabComponent,
            inputs: { profile },
        }
    }

    getSuggestedName (profile: SSHProfile): string {
        return `${profile.options.user}@${profile.options.host}:${profile.options.port}`
    }

    getDescription (profile: PartialProfile<SSHProfile>): string {
        return profile.options?.host ?? ''
    }

    deleteProfile (profile: SSHProfile): void {
        this.passwordStorage.deletePassword(profile)
    }

    quickConnect (query: string): PartialProfile<SSHProfile> {
        let user = 'root'
        let host = query
        let port = 22
        if (host.includes('@')) {
            const parts = host.split(/@/g)
            host = parts[parts.length - 1]
            user = parts.slice(0, parts.length - 1).join('@')
        }
        if (host.includes('[')) {
            port = parseInt(host.split(']')[1].substring(1))
            host = host.split(']')[0].substring(1)
        } else if (host.includes(':')) {
            port = parseInt(host.split(/:/g)[1])
            host = host.split(':')[0]
        }

        return {
            name: query,
            type: 'ssh',
            options: {
                host,
                user,
                port,
            },
        }
    }
}
""",
    'verilog': """/*******************************************************************************    
 *
 * Copyright(C) 2017 ERC CISST, Johns Hopkins University.
 *
 * Module:  EncCtrl_tb
 *
 * Purpose: This is a test bench for encoder velocity estimation. It generates
 * quadrature signals of varying lengths in sine_wave_gen (currently sine wave 
 * is approximated as 0s and 1s only) to simulate encoders at different velocities.
 *
 * Revision history
 *     12/24/2017    Jie Ying Wu        Initial commit
 */

 `timescale 1ns / 1ps

 module EncCtrl_tb;

  reg         clk1394;
  wire        clk_fast;
  reg         clk_encoder;
  wire  [8:0] x;
  integer     i;
  reg   [8:0] speed [0:9];
  wire        a;
  wire        b;
  reg         dir;
  wire [31:0] period;
  wire [31:0] acc;

  // Define encoder cycle length
  assign cycle = 20;

  initial begin
        // Initialize Inputs
        clk_encoder = 0;
        i       = 0;
        clk1394 = 0;
        dir     = 0;
        speed[0] = 250;
        speed[1] = 200;
        speed[2] = 150;
        speed[3] = 100;
        speed[4] = 50;
        speed[5] = 100;
        speed[6] = 150;
        speed[7] = 200;
        speed[8] = 250;
        speed[9] = 300;
    end

    // generate clk
    always begin
        #1 clk1394 <= ~clk1394;     // system clock 
    end

    always begin
        #x clk_encoder <= ~clk_encoder;     // system clock 
    end

    always begin
        #50000 i <= (i == 9) ? 0 : i + 1;
    end

    assign x = speed[i];

    sine_wave_gen uut (
        .Clk(clk_encoder), 
        .a(a),
        .b(b)
    );

ClkDiv divenc1(clk1394, clk_fast); defparam divenc1.width = 1;   // 49.152 MHz / 2**4 ==> 3.072 MHz

EncPeriod VelEstimate(
    .clk(clk1394),       // sysclk
    .clk_fast(clk_fast), // count this clock between encoder ticks
    .reset(1),           // global reset signal
    .a(a),               // quad encoder line a
    .b(b),               // quad encoder line b
    .dir(dir),           // dir from EncQuad
    .period(period),      // num of fast clock ticks
    .acc(acc)
);

endmodule


module sine_wave_gen(Clk,a,b);
//declare input and output
    input Clk;
    output a;
    output b;
//declare the sine ROM - 30 registers each 8 bit wide.  
    reg [7:0] sine [0:29];
//Internal signals  
    integer i;  
    integer j;

//Initialize the sine rom with samples. 
    initial begin
        i = 0;
        j = 8;
        sine[0] = 1;
        sine[1] = 1;
        sine[2] = 1;
        sine[3] = 1;
        sine[4] = 1;
        sine[5] = 1;
        sine[6] = 1;
        sine[7] = 1;
        sine[8] = 1;
        sine[9] = 1;
        sine[10] = 1;
        sine[11] = 1;
        sine[12] = 1;
        sine[13] = 1;
        sine[14] = 1;
        sine[15] = 0;
        sine[16] = 0;
        sine[17] = 0;
        sine[18] = 0;
        sine[19] = 0;
        sine[20] = 0;
        sine[21] = 0;
        sine[22] = 0;
        sine[23] = 0;
        sine[24] = 0;
        sine[25] = 0;
        sine[26] = 0;
        sine[27] = 0;
        sine[28] = 0;
        sine[29] = 0;
    end

    assign a = (sine[i]);
    assign b = (sine[j]);

    //At every positive edge of the clock, output a sine wave sample.
    always@ (posedge(Clk))
    begin
        i <= i+ 1;
        if(i == 29)
            i <= 0;
        j <= j+ 1;
        if(j == 29)
            j <= 0;
    end
endmodule
""",
 }
JAVA = SAMPLE_CODE["java"]

LONG_JAVA = """/*
 * ====================================================================
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 * ====================================================================
 *
 * This software consists of voluntary contributions made by many
 * individuals on behalf of the Apache Software Foundation.  For more
 * information on the Apache Software Foundation, please see
 * <http://www.apache.org/>.
 *
 */

package ch.boye.httpclientandroidlib.impl.cookie;

import java.util.ArrayList;
import java.util.List;

import ch.boye.httpclientandroidlib.FormattedHeader;
import ch.boye.httpclientandroidlib.Header;
import ch.boye.httpclientandroidlib.HeaderElement;
import ch.boye.httpclientandroidlib.annotation.NotThreadSafe;
import ch.boye.httpclientandroidlib.cookie.ClientCookie;
import ch.boye.httpclientandroidlib.cookie.Cookie;
import ch.boye.httpclientandroidlib.cookie.CookieOrigin;
import ch.boye.httpclientandroidlib.cookie.MalformedCookieException;
import ch.boye.httpclientandroidlib.cookie.SM;
import ch.boye.httpclientandroidlib.message.BufferedHeader;
import ch.boye.httpclientandroidlib.message.ParserCursor;
import ch.boye.httpclientandroidlib.util.Args;
import ch.boye.httpclientandroidlib.util.CharArrayBuffer;

/**
 * This {@link ch.boye.httpclientandroidlib.cookie.CookieSpec} implementation conforms to
 * the original draft specification published by Netscape Communications.
 * It should be avoided unless absolutely necessary for compatibility with
 * legacy applications.
 *
 * @since 4.0
 */
@NotThreadSafe // superclass is @NotThreadSafe
public class NetscapeDraftSpec extends CookieSpecBase {

    protected static final String EXPIRES_PATTERN = "EEE, dd-MMM-yy HH:mm:ss z";

    private final String[] datepatterns;

    /** Default constructor */
    public NetscapeDraftSpec(final String[] datepatterns) {
        super();
        if (datepatterns != null) {
            this.datepatterns = datepatterns.clone();
        } else {
            this.datepatterns = new String[] { EXPIRES_PATTERN };
        }
        registerAttribHandler(ClientCookie.PATH_ATTR, new BasicPathHandler());
        registerAttribHandler(ClientCookie.DOMAIN_ATTR, new NetscapeDomainHandler());
        registerAttribHandler(ClientCookie.MAX_AGE_ATTR, new BasicMaxAgeHandler());
        registerAttribHandler(ClientCookie.SECURE_ATTR, new BasicSecureHandler());
        registerAttribHandler(ClientCookie.COMMENT_ATTR, new BasicCommentHandler());
        registerAttribHandler(ClientCookie.EXPIRES_ATTR, new BasicExpiresHandler(
                this.datepatterns));
    }

    /** Default constructor */
    public NetscapeDraftSpec() {
        this(null);
    }

    /**
      * Parses the Set-Cookie value into an array of <tt>Cookie</tt>s.
      *
      * <p>Syntax of the Set-Cookie HTTP Response Header:</p>
      *
      * <p>This is the format a CGI script would use to add to
      * the HTTP headers a new piece of data which is to be stored by
      * the client for later retrieval.</p>
      *
      * <PRE>
      *  Set-Cookie: NAME=VALUE; expires=DATE; path=PATH; domain=DOMAIN_NAME; secure
      * </PRE>
      *
      * <p>Please note that the Netscape draft specification does not fully conform to the HTTP
      * header format. Comma character if present in <code>Set-Cookie</code> will not be treated
      * as a header element separator</p>
      *
      * @see <a href="http://web.archive.org/web/20020803110822/http://wp.netscape.com/newsref/std/cookie_spec.html">
      *  The Cookie Spec.</a>
      *
      * @param header the <tt>Set-Cookie</tt> received from the server
      * @return an array of <tt>Cookie</tt>s parsed from the Set-Cookie value
      * @throws MalformedCookieException if an exception occurs during parsing
      */
    public List<Cookie> parse(final Header header, final CookieOrigin origin)
            throws MalformedCookieException {
        Args.notNull(header, "Header");
        Args.notNull(origin, "Cookie origin");
        if (!header.getName().equalsIgnoreCase(SM.SET_COOKIE)) {
            throw new MalformedCookieException("Unrecognized cookie header '"
                    + header.toString() + "'");
        }
        final NetscapeDraftHeaderParser parser = NetscapeDraftHeaderParser.DEFAULT;
        final CharArrayBuffer buffer;
        final ParserCursor cursor;
        if (header instanceof FormattedHeader) {
            buffer = ((FormattedHeader) header).getBuffer();
            cursor = new ParserCursor(
                    ((FormattedHeader) header).getValuePos(),
                    buffer.length());
        } else {
            final String s = header.getValue();
            if (s == null) {
                throw new MalformedCookieException("Header value is null");
            }
            buffer = new CharArrayBuffer(s.length());
            buffer.append(s);
            cursor = new ParserCursor(0, buffer.length());
        }
        return parse(new HeaderElement[] { parser.parseHeader(buffer, cursor) }, origin);
    }

    public List<Header> formatCookies(final List<Cookie> cookies) {
        Args.notEmpty(cookies, "List of cookies");
        final CharArrayBuffer buffer = new CharArrayBuffer(20 * cookies.size());
        buffer.append(SM.COOKIE);
        buffer.append(": ");
        for (int i = 0; i < cookies.size(); i++) {
            final Cookie cookie = cookies.get(i);
            if (i > 0) {
                buffer.append("; ");
            }
            buffer.append(cookie.getName());
            final String s = cookie.getValue();
            if (s != null) {
                buffer.append("=");
                buffer.append(s);
            }
        }
        final List<Header> headers = new ArrayList<Header>(1);
        headers.add(new BufferedHeader(buffer));
        return headers;
    }

    public int getVersion() {
        return 0;
    }

    public Header getVersionHeader() {
        return null;
    }

    /**
     * This {@link ch.boye.httpclientandroidlib.cookie.CookieSpec} implementation conforms to
     * the original draft specification published by Netscape Communications.
     * It should be avoided unless absolutely necessary for compatibility with
     * legacy applications.
     *
     * @since 4.0
     */
    @Override
    public String toString() {
        return "netscape";
    }

}"""


class TestCodeParser(unittest.TestCase):

    def test_parsing_all(self):
        parser = CodeParser()

        for language, code in SAMPLE_CODE.items():
            with self.subTest(f"Parsing {language}") as t:
                tree = parser.parse(code, language)
                decoded_code = parser.unparse(tree)
                if code != decoded_code:
                    print(code)
                    print()
                    print(decoded_code)
                    tree.pprint()
                    print('#' * 100)
                self.assertSequenceEqual(code, decoded_code)


    def test_parsing_css(self):
        parser = CodeParser()
        language = "css"

        code = SAMPLE_CODE[language]
        tree = parser.parse(code, language)
        decoded_code = parser.unparse(tree)
        if code != decoded_code:
            print(code)
            print()
            print(decoded_code)
            tree.pprint()
            print('#' * 100)
        self.assertSequenceEqual(code, decoded_code)


    def test_parsing_java(self):
        parser = CodeParser()
        language = "java"

        code = SAMPLE_CODE[language]
        tree = parser.parse(code, language)
        decoded_code = parser.unparse(tree)
        if code != decoded_code:
            print(code)
            print()
            print(decoded_code)
            tree.pprint()
            print('#' * 100)
        self.assertSequenceEqual(code, decoded_code)


class TestBPE(unittest.TestCase):

    def test_bpe_decoding(self):
        parser = CodeParser()

        tree = parser.parse(JAVA, "java")
        # tree = tree[1].detach()
        tree.pprint()

        bpe = TreeBPE("assets/bpe.model")
        bpe_tree = bpe.encode(tree)
        bpe_tree.pprint()

        decoded_tree = bpe.decode(bpe_tree)
        decoded_tree.pprint()

        self.assertSequenceEqual(tree.node_data, decoded_tree.node_data)
        self.assertSequenceEqual(tree.descendants.tolist(), decoded_tree.descendants.tolist())
        self.assertSequenceEqual(tree.parents.tolist(), decoded_tree.parents.tolist())

    def test_tokenizer_bpe_decoding(self):
        parser = CodeParser()

        tree = parser.parse(JAVA, "java")
        # tree = tree[1].detach()
        tree.pprint()

        bpe = WordPieceBPE("/home/johannes/projects/tokenizecode/tokenizecode_cli/bpemodel.json")
        bpe_tree = bpe.encode(tree)
        bpe_tree.pprint()

        decoded_tree = bpe.decode(bpe_tree)
        decoded_tree.pprint()

        self.assertSequenceEqual(tree.node_data, decoded_tree.node_data)
        self.assertSequenceEqual(tree.descendants.tolist(), decoded_tree.descendants.tolist())
        self.assertSequenceEqual(tree.parents.tolist(), decoded_tree.parents.tolist())


class TestTokenizer(unittest.TestCase):

    def test_detokenize(self):
        bpe = TreeBPE("assets/bpe.model")
        tokenizer = CodeTokenizer(bpe)

        tree = tokenizer.tokenize(JAVA, "java")
        tree.pprint()
        code = tokenizer.detokenize(tree)

        self.assertSequenceEqual(JAVA, code)

    def test_detokenize_splitlines(self):
        bpe = TreeBPE("assets/bpe.model")
        tokenizer = CodeTokenizer(bpe, parser=CodeParser(traversal=SplitLinesTraversal()))

        tree = tokenizer.tokenize(LONG_JAVA, "java")
        tree.pprint()
        code = tokenizer.detokenize(tree)

        self.assertSequenceEqual(LONG_JAVA, code)

if __name__ == '__main__':
    unittest.main()
