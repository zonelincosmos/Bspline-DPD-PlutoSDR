% B-spline DPD on ADALM-Pluto SDR  (hardware-in-loop)
%
% Same TX / RX / Pluto setup, waveform, and TX->RX sync as the companion
% project ILC-DPD-PlutoSDR (https://github.com/zonelincosmos/ILC-DPD-PlutoSDR).
% Only the predistorter differs: the DPD here is a complex gain G(|x|) on a
% cubic B-spline basis, WLS-fit, with an additive ILC-style coefficient
% update and the same I/Q averaging the ILC script uses.
%
%   xd          = G(|x|) .* x                       (B-spline LUT predistort)
%   y           = average of N_avg PA(xd) captures   (de-noised, RMS-aligned)
%   c_{k+1}     = c_k + mu * (AtWA \ AtWb)            (additive WLS / ILC)
%
% B-spline basis via the de Boor recursion; sample-by-sample normal-matrix
% WLS accumulation.
%
% License: MIT (see LICENSE).

%% B-spline DPD, hardware-in-loop on Pluto SDR
clear all;clc;close all

%% -- load waveform (same pattern as ILC) --
load('RefSignal.mat')
RefSignalX = RefSignal(:);

%% -- Setup Pluto SDR (identical to the companion ILC-DPD-PlutoSDR) --
CenterFrequency    = 2000; % MHz
BasebandSampleRate = 1;  % MHz
TxGain = -3;
RxGain =  0;

x    = RefSignalX;
N    = length(x);
r    = abs(x);
pad          = 1000;
tx_len       = N + 2*pad;
n_frames_mul = 4;                  % capture 4 periods (sync to clean middle copy)
frameSize    = tx_len * n_frames_mul;

rx = sdrrx('Pluto', ...
    'CenterFrequency',    (CenterFrequency) * 1e6, ...
    'SamplesPerFrame',    frameSize, ...
    'OutputDataType',     'double', ...
    'BasebandSampleRate', BasebandSampleRate * 1e6, ...
    'EnableBurstMode',    false, ...
    'GainSource',         'Manual', ...
    'Gain',               RxGain);
disp(info(rx));

tx = sdrtx('Pluto', ...
    'CenterFrequency',    (CenterFrequency) * 1e6, ...
    'Gain',              TxGain, ...
    'BasebandSampleRate', BasebandSampleRate * 1e6);

fprintf('Signal N = %d, TX len = %d, frameSize = %d\n', N, tx_len, frameSize);

%% Cubic Lagrange Farrow coefficient matrix (fractional resync).
Cfarrow = [ -1/6,  1/2, -1/2,  1/6;
             1/2, -1,    1/2,  0;
            -1/3, -1/2,  1,   -1/6;
             0,    1,    0,    0   ];

%% -- B-spline DPD parameters --
deg   = 3;
nSeg  = 16;                            % split 0~1 into 16 segments
t_brk = (0:1/nSeg:1)';                 % 17 break points
t_kv  = [repmat(t_brk(1),deg,1); t_brk; repmat(t_brk(end),deg,1)];
K     = length(t_kv) - deg - 1;        % = 19

lam      = 1e-7;
w_alpha  = 20;
w_pow    = 4;
G_target = 1.0;
G_max    = 10^(4/20);            % DPD gain ceiling: strict +4 dB (only cap in use)
r_drive  = 1.0;                  % keep |xd| within Pluto DAC full-scale
mu       = 0.5;                  % ILC learning rate (same as ILC script)
N_avg    = 16;                   % I/Q averaged captures per iter 
n_iter   = 10;                   % same as ILC script

% LUT basis built with the SAME evaluator (bspline_eval4) the WLS uses, so
% applyLUT and the coefficient update share one basis.  N_LUT = nSeg: one
% LUT entry per B-spline segment (no oversampling).
N_LUT      = 16;
r_lut      = linspace(0,1,N_LUT)';
B_lut_base = buildBspl(r_lut, t_kv, deg, K);

AtWA_full = wls_build_AtWA(x, w_alpha, w_pow, t_kv, deg, K) + lam * eye(K);

%% -- Baseline capture (no DPD) --
fprintf('\n--- Baseline capture (no DPD) ---\n');
[y0, soPrev] = captureAndSync(x, rx, tx, RefSignalX, pad, frameSize, Cfarrow, []);
NMSE0 = 10*log10(mean(abs(y0 - G_target*x).^2) / mean(abs(G_target*x).^2));
fprintf('Baseline NMSE = %.2f dB\n', NMSE0);

% Observed saturation ceiling (refined from each capture, like the reference)
Psat  = max(abs(y0)) * 0.99;
r_sat = Psat / G_target;

%% -- Identity seed: G_spline(r) = 1, basis-consistent weighted LS --
% Full-rank (K unknowns, N equations) and uses the same B-spline basis as
% the WLS update, so iteration 1 transmits xd ~ x (pass-through).
Bx = buildBspl(r, t_kv, deg, K);
wv = 1 + w_alpha * r.^w_pow;
c  = (Bx.' * (wv .* Bx) + lam*eye(K)) \ (Bx.' * (wv .* ones(N,1)));

%% -- B-spline DPD: hardware in-loop iterations --
nmse    = zeros(n_iter,1);
lut_log = zeros(N_LUT, n_iter);   % per-iteration LUT snapshots (for dB plot)
y       = y0;
for k = 1:n_iter
    % (a) Capped LUT -> predistorted reference
    lut_c = capLUT(B_lut_base * c, G_max, r_lut, [], []);   % only G_max cap; drive cap (Step 2) + extrapolation (Step 3) disabled
    % Flatten LUT endpoints (manual jitter suppression at r=0 and r=1)
    lut_c(1)     = lut_c(2);
    lut_c(end)   = lut_c(end-2);
    lut_c(end-1) = lut_c(end-2);
    lut_log(:,k) = lut_c;   % snapshot (post-paste) for end-of-run LUT (dB) plot
    xd    = applyLUT(x, lut_c, r_lut);

    % (b) I/Q averaged capture of PA(xd)
    Msum = zeros(N,1);
    for j = 1:N_avg
        [yj, soPrev] = captureAndSync(xd, rx, tx, RefSignalX, pad, frameSize, Cfarrow, soPrev);
        Msum = Msum + yj;
    end
    y = Msum / N_avg;

    % (c) Refine observed saturation ceiling
    Psat  = max(Psat, max(abs(y)) * 0.99);
    r_sat = Psat / G_target;

    % (d) Measured NMSE of current predistorter
    nmse(k) = 10*log10(mean(abs(y - G_target*x).^2) / mean(abs(G_target*x).^2));
    fprintf('Iter %d  NMSE = %.2f dB   max|xd| = %.3f\n', k, nmse(k), max(abs(xd)));

    % (e) Additive WLS / ILC coefficient update
    AtWb = wls_build_AtWb(x, y, Psat, G_target, w_alpha, w_pow, t_kv, deg, K);
    c    = c + mu * (AtWA_full \ AtWb);

    % (f) Per-iteration sync / linearization check
    fh = figure('Visible','off','Position',[100 100 1100 700]);
    subplot(2,1,1);
    plot(abs(RefSignalX)); hold on; plot(abs(y));
    legend('|RefSignalX|','|y|'); grid on;
    title(sprintf('B-spline iter %d  NMSE=%.2f dB  max|xd|=%.3f', k, nmse(k), max(abs(xd))));
    subplot(2,1,2);
    plot(abs(x), abs(y), '.'); hold on; plot(abs(x), abs(x), 'k-');
    grid on; xlabel('|RefSignalX|'); ylabel('|y|'); title('AM-AM');
    exportgraphics(fh, sprintf('bspline_iter_%02d.png', k));
    close(fh);
end
y_final = y;

%% -- Release hardware --
release(rx);
release(tx);

%% -- Results --
figure;
plot(0:n_iter, [NMSE0; nmse], '-o'); grid on;
xlabel('Iteration'); ylabel('NMSE (dB)');
title('B-spline DPD convergence');

figure;
plot(abs(x), abs(y0),      '.'); hold on;
plot(abs(x), abs(y_final), '.');
plot(abs(x), abs(x),       'k-');
grid on; xlabel('|RefSignalX|'); ylabel('|CompressedSignalY|');
legend('Before DPD','After DPD','Ideal','Location','NorthWest');
title('B-spline DPD  AM-AM');

figure;
plot(r_lut, 20*log10(abs(lut_log)), '-o'); grid on;
xlabel('|RefSignalX|'); ylabel('|G_{DPD}| (dB)');
title(sprintf('B-spline DPD LUT magnitude (dB), all %d iterations overlaid', n_iter));

sa1 = dsp.SpectrumAnalyzer('SampleRate',BasebandSampleRate*1e6,'SpectralAverages',5, ...
    'ShowLegend',true,'ChannelNames',{'RefSignalX','Without B-spline DPD','With B-spline DPD'});
sa1.YLimits = [-10,100];
sa1([RefSignalX(:), y0(:), y_final(:)]);

%%
% =============================================================================
% Local functions
% =============================================================================

function [CompressedSignalY, SyncOffset] = captureAndSync(s, rx, tx, RefSignalX, pad, frameSize, Cfarrow, soPrev)
% Transmit zero-padded s via Pluto, capture one frame, integer (xcorr) +
% fractional (Farrow) sync, return DC-removed RMS-normalized CompressedSignalY (Nx1).
% soPrev: previous SyncOffset for tracking ([] = full search on baseline).
    txWave = complex([zeros(pad,1); s(:); zeros(pad,1)]);
    release(tx);
    tx.transmitRepeat(txWave);

    for i = 1:2, rx(); end                 % warm-up: discard settling frames
    [d, ~, of] = rx();
    if of, fprintf('  (RX overflow)\n'); end
    dataVec = d(:);

    % --- Integer sync: collect candidate xcorr peaks (Farrow-safe range) ---
    [corr_vals, lags] = xcorr(dataVec, RefSignalX);
    cv  = abs(corr_vals);
    so  = lags + 1;
    Nr     = length(RefSignalX);
    period = 2*pad + numel(s);
    valid = (so >= period + 2) & (so <= 3*period - Nr - 2);
    cv(~valid) = -inf;
    gmax = max(cv);
    cand = [];
    for p = 1:5
        [pk, ii] = max(cv);
        if pk < 0.6 * gmax, break; end
        cand(end+1) = so(ii);              %#ok<AGROW>
        lo = max(1, ii - round(Nr/2));
        hi = min(numel(cv), ii + round(Nr/2));
        cv(lo:hi) = -inf;
    end

    % --- Pick (SyncOffset, FracDelay) minimising post-Farrow envelope error ---
    n      = (0:Nr-1).';
    RmsX0  = rms(RefSignalX);
    absRef = abs(RefSignalX);
    muGrid = (-1:0.002:1).';
    bestErr = inf; SyncOffset = cand(1); FracDelay = 0;
    for ci = 1:numel(cand)
        so_c = cand(ci);
        for kk = 1:numel(muGrid)
            yk = farrowEval(dataVec, so_c, n, muGrid(kk), Cfarrow);
            yk = yk - mean(yk);
            yk = yk * (RmsX0 / rms(yk));
            e  = norm(absRef - abs(yk));
            if e < bestErr
                bestErr = e; SyncOffset = so_c; FracDelay = muGrid(kk);
            end
        end
    end
    fprintf('  SyncOffset = %d, FracDelay = %.4f, err = %.4g\n', ...
            SyncOffset, FracDelay, bestErr);

    CompressedSignalY = farrowEval(dataVec, SyncOffset, n, FracDelay, Cfarrow);
    CompressedSignalY = CompressedSignalY - mean(CompressedSignalY);              % Remove DC
    CompressedSignalY = CompressedSignalY * (rms(RefSignalX) / rms(CompressedSignalY)); % RMS normalize to RefSignalX scale
    phi  = angle(CompressedSignalY' * RefSignalX);            % constant loopback phase rotation
    CompressedSignalY = CompressedSignalY * exp(1j*phi);             % de-rotate to align with RefSignalX
end

% --- Cubic Farrow fractional resampler ---
function y = farrowEval(dataVec, SyncOffset, n, mu, Cfarrow)
    di = floor(mu);
    f  = mu - di;
    m  = SyncOffset + n + di;
    X  = [dataVec(m-1), dataVec(m), dataVec(m+1), dataVec(m+2)];
    P  = X * Cfarrow.';
    y  = ((P(:,1)*f + P(:,2))*f + P(:,3))*f + P(:,4);
end

function lut = capLUT(lut, G_max, r_lut, r_drive, r_sat)
% Cap LUT gain and extrapolate data-sparse region.  Three-step process:
%
%   Step 1  Gain cap     : |G(r)| <= G_max              (HW gain ceiling)
%   Step 2  Drive cap    : |G(r)*r| <= r_drive           (PA overdrive protection)
%   Step 3  Extrapolation: r > r_sat -> blend to ideal    (data-sparse region)

    % Step 1+2: per-node gain ceiling
    g_lim = G_max * ones(size(lut));
    if nargin >= 4 && ~isempty(r_drive)
        g_lim = min(g_lim, r_drive ./ max(r_lut, eps));
    end
    mag  = abs(lut);
    over = mag > g_lim;
    lut(over) = lut(over) ./ mag(over) .* g_lim(over);

    % Step 3: smooth interpolation from B-spline anchor to ideal saturation gain
    if nargin >= 5 && ~isempty(r_sat) && ~isempty(r_lut)
        i_sat = find(r_lut <= r_sat, 1, 'last');
        if ~isempty(i_sat) && i_sat >= 2 && i_sat < length(lut)
            g_anchor  = abs(lut(i_sat));
            ph_anchor = angle(lut(i_sat));
            span      = r_lut(end) - r_lut(i_sat);
            for i = (i_sat+1):length(lut)
                frac  = (r_lut(i) - r_lut(i_sat)) / max(span, eps);
                g_new = g_anchor * (1 - frac) + g_lim(i) * frac;
                lut(i) = g_new * exp(1j * ph_anchor);
            end
        end
    end
end

function xd = applyLUT(x, lut, r_lut)
% LUT-based DPD compensator: linear interpolation of complex gain G(r).
    r    = abs(x);
    dr   = r_lut(2) - r_lut(1);
    r_c  = min(r, r_lut(end));
    t    = r_c / dr;
    i0   = min(floor(t), length(r_lut)-1);
    i_lo = i0 + 1;
    i_hi = min(i_lo + 1, length(r_lut));
    frac = t - double(i0);
    G    = (1 - frac) .* lut(i_lo) + frac .* lut(i_hi);
    xd   = G .* x;
end

function B = buildBspl(r_nodes, t_kv, deg, K)
% Evaluate all K cubic B-spline basis functions at each node, using the
% SAME de Boor evaluator (bspline_eval4) the WLS accumulation uses.  This
% keeps applyLUT and the coefficient update on one consistent basis
% (including the r = 1 endpoint, where the old buildB/bspF disagreed).
    r_nodes = r_nodes(:);
    B = zeros(numel(r_nodes), K);
    for ii = 1:numel(r_nodes)
        [vals, first_k] = bspline_eval4(r_nodes(ii), t_kv, deg);
        B(ii, first_k:first_k+3) = vals(:).';
    end
end

function [vals, first_k] = bspline_eval4(r_val, t_kv, deg)
% BSPLINE_EVAL4  De Boor triangle evaluation of the cubic B-spline basis.
% Returns the 4 nonzero basis values and the 1-based first_k.

    K    = length(t_kv) - deg - 1;
    nSeg = length(t_kv) - 2*deg - 1;   % # uniform segments over [0,1]

    % Clamp r to [0, 1]
    r_val = max(0, min(1, r_val));

    % --- Find knot span (0-based), derived from the knot vector ---
    if r_val >= 1.0
        span_0 = K - 1;
    else
        span_0 = deg + floor(r_val * nSeg);   % segment width = 1/nSeg
        if span_0 > K - 1, span_0 = K - 1; end
    end

    first_k_0 = span_0 - deg;   % 0-based first nonzero basis index

    % --- de Boor triangle ---
    N0 = zeros(deg + 2, 1);   % extra element for safety
    N0(1) = 1.0;

    for j = 1:deg
        saved = 0.0;
        for s = 0:(j-1)
            left_idx_0  = span_0 - j + 1 + s;
            right_idx_0 = span_0 + 1 + s;

            d1 = t_kv(right_idx_0 + 1) - t_kv(left_idx_0 + 1);
            if d1 > 1e-12
                temp = N0(s + 1) / d1;
            else
                temp = 0.0;
            end

            N0(s + 1) = saved + (t_kv(right_idx_0 + 1) - r_val) * temp;
            saved = (r_val - t_kv(left_idx_0 + 1)) * temp;
        end
        N0(j + 1) = saved;
    end

    % Output: 4 values in N0(1:4), 1-based first_k
    vals = N0(1:4);
    first_k = first_k_0 + 1;   % convert to 1-based

    % Clamp first_k
    if first_k < 1, first_k = 1; end
    if first_k + 3 > K, first_k = K - 3; end
end

function AtWA = wls_build_AtWA(x, w_alpha, w_pow, t_kv, deg, K)
% WLS_BUILD_ATWA  Accumulate the K x K weighted normal matrix sample-by-sample.
%   AtWA = sum_n  w_n * conj(phi_n) * phi_n.'

    AtWA = zeros(K, K);
    N_samp = length(x);

    for n = 1:N_samp
        rv = abs(x(n));
        [vals, first_k] = bspline_eval4(rv, t_kv, deg);

        phi = vals * x(n);

        w  = 1 + w_alpha * rv^w_pow;
        sw = sqrt(w);
        phi_sw = phi * sw;

        idx = first_k : (first_k + 3);
        AtWA(idx, idx) = AtWA(idx, idx) + conj(phi_sw) * phi_sw.';
    end
end

function AtWb = wls_build_AtWb(x, y, Psat, G_target, w_alpha, w_pow, t_kv, deg, K)
% WLS_BUILD_ATWB  Accumulate the K x 1 weighted right-hand side sample-by-sample.
%   AtWb = sum_n  w_n * conj(phi_n) * (target_n - y_n)

    AtWb = zeros(K, 1);
    N_samp = length(x);

    for n = 1:N_samp
        rv = abs(x(n));
        [vals, first_k] = bspline_eval4(rv, t_kv, deg);

        phi = vals * x(n);

        w  = 1 + w_alpha * rv^w_pow;
        sw = sqrt(w);
        phi_sw = phi * sw;

        r_tgt = min(G_target * rv, Psat);
        x_tgt = r_tgt / max(rv, 1e-12) * x(n);

        err_sw = (x_tgt - y(n)) * sw;

        idx = first_k : (first_k + 3);
        AtWb(idx) = AtWb(idx) + conj(phi_sw) * err_sw;
    end
end
