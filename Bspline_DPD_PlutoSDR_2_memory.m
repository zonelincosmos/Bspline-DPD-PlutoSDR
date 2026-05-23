% Memory B-spline DPD on ADALM-Pluto SDR (hardware-in-loop)
%
% Extension of Bspline_DPD_PlutoSDR_2.m to handle PA memory effects.
% Same TX/RX/Pluto/sync infrastructure; only the predistorter math is
% extended from a single memoryless LUT to a sum of 5 memory taps:
%
%   xd(n) = sum_i  x(n - m1_i) * G_i(|x(n - m2_i)|)
%
%   Term i   (m1_i, m2_i)        Type
%   -------------------------------------------------
%     1      (0, 0)               self : x(n)   * G_1(|x(n)|)
%     2      (1, 1)               self : x(n-1) * G_2(|x(n-1)|)
%     3      (2, 2)               self : x(n-2) * G_3(|x(n-2)|)
%     4      (0, 1)               cross: x(n)   * G_4(|x(n-1)|)
%     5      (1, 2)               cross: x(n-1) * G_5(|x(n-2)|)
%
% Each G_i is a complex cubic B-spline LUT (same basis as the original
% script) -- this is the "Memory B-spline" decomposition:
%
%   y(n) = sum_m x(n-m1) * [ sum_j c_{m,j} * B_j(|x(n-m2)|) ]
%                          \__________________________________/
%                                 per-tap complex LUT
%
% The whole 5*K coefficient vector is fitted by a single WLS solve per
% ILC iteration, with sample-by-sample normal-matrix accumulation.
%
% License: MIT.

%% Memory B-spline DPD, hardware-in-loop on Pluto SDR
clear all; clc; close all  %#ok<CLALL>

%% -- load waveform (same as original) --
load('RefSignal.mat')
RefSignalX = RefSignal(:);

%% -- Setup Pluto SDR (identical to Bspline_DPD_PlutoSDR_2.m) --
CenterFrequency    = 2000; % MHz
BasebandSampleRate = 1;    % MHz
TxGain = -3;
RxGain =  0;

x    = RefSignalX;
N    = length(x);
r    = abs(x);
pad          = 1000;
tx_len       = N + 2*pad;
n_frames_mul = 4;
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

%% -- Memory B-spline configuration (5 taps as specified) --
% Each term contributes  G_i( |x(n-m2)| ) * x(n-m1)  to xd(n).
terms = struct('m1', {0, 1, 2, 0, 1}, ...
               'm2', {0, 1, 2, 1, 2}, ...
               'type', {'self', 'self', 'self', 'cross', 'cross'});
T_terms = length(terms);
M_max   = max([terms.m1, terms.m2]);
fprintf('Memory B-spline: %d terms, memory depth M = %d\n', T_terms, M_max);

%% -- B-spline basis params (same as original) --
deg   = 3;
nSeg  = 16;
t_brk = (0:1/nSeg:1)';
t_kv  = [repmat(t_brk(1),deg,1); t_brk; repmat(t_brk(end),deg,1)];
K     = length(t_kv) - deg - 1;
total_K = K * T_terms;

lam      = 1e-7;
lam_mem  = 1e-6;            % slightly more regularization for 5*K params
w_alpha  = 20;
w_pow    = 4;
G_target = 1.0;
G_max    = 10^(4/20);       % self-term DPD gain ceiling: +4 dB
r_drive  = 1.0;             % keep |xd| within Pluto DAC full-scale
mu       = 0.3;             % ILC learning rate (slightly < 0.5 due to more params)
N_avg    = 16;
n_iter   = 10;

N_LUT      = 16;
r_lut      = linspace(0,1,N_LUT)';
B_lut_base = buildBspl(r_lut, t_kv, deg, K);

% Pre-build constant AtWA on x (depends only on the input signal, not iter)
AtWA_full = wls_build_AtWA_memory(x, w_alpha, w_pow, t_kv, deg, K, terms) + ...
            lam_mem * eye(total_K);

%% -- Baseline capture (no DPD) --
fprintf('\n--- Baseline capture (no DPD) ---\n');
[y0, soPrev] = captureAndSync(x, rx, tx, RefSignalX, pad, frameSize, Cfarrow, []);
NMSE0 = 10*log10(mean(abs(y0 - G_target*x).^2) / mean(abs(G_target*x).^2));
fprintf('Baseline NMSE = %.2f dB\n', NMSE0);

Psat  = max(abs(y0)) * 0.99;
r_sat = Psat / G_target;

%% -- Identity seed: self term 1 starts at G=1; cross terms start at 0 --
c_matrix = zeros(K, T_terms);   % K x 5 complex coefficient matrix
% Term 1 (self at n) -> identity LUT G(r) = 1
Bx = buildBspl(r, t_kv, deg, K);
wv = 1 + w_alpha * r.^w_pow;
c_matrix(:, 1) = (Bx.' * (wv .* Bx) + lam*eye(K)) \ (Bx.' * (wv .* ones(N,1)));

%% -- Memory B-spline DPD: hardware-in-loop iterations --
nmse    = zeros(n_iter,1);
lut_log = zeros(N_LUT, T_terms, n_iter);   % per-iter, per-term LUT snapshots
y       = y0;
for k = 1:n_iter
    % (a) Build per-term LUTs from coefficient columns
    luts = cell(T_terms, 1);
    for i_t = 1:T_terms
        lut_i = B_lut_base * c_matrix(:, i_t);
        if strcmp(terms(i_t).type, 'self')   % cap self terms (identity baseline)
            lut_i = capLUT(lut_i, G_max, r_lut, [], []);
        end
        % Flatten endpoints (jitter suppression at r=0, r=1)
        lut_i(1)     = lut_i(2);
        lut_i(end)   = lut_i(end-2);
        lut_i(end-1) = lut_i(end-2);
        luts{i_t}        = lut_i;
        lut_log(:,i_t,k) = lut_i;
    end

    % (b) Apply memory DPD
    xd = applyMemoryLUT(x, luts, terms, r_lut);
    % Cap drive
    mag_xd = abs(xd);
    over   = mag_xd > r_drive;
    xd(over) = xd(over) ./ mag_xd(over) * r_drive;

    % (c) I/Q averaged capture of PA(xd)
    Msum = zeros(N,1);
    for j = 1:N_avg
        [yj, soPrev] = captureAndSync(xd, rx, tx, RefSignalX, pad, frameSize, Cfarrow, soPrev);
        Msum = Msum + yj;
    end
    y = Msum / N_avg;

    % (d) Refine observed saturation ceiling
    Psat  = max(Psat, max(abs(y)) * 0.99);
    r_sat = Psat / G_target;

    % (e) NMSE
    nmse(k) = 10*log10(mean(abs(y - G_target*x).^2) / mean(abs(G_target*x).^2));
    fprintf('Iter %d  NMSE = %.2f dB   max|xd| = %.3f\n', k, nmse(k), max(abs(xd)));

    % (f) WLS / ILC coefficient update (single solve for all 5*K params)
    AtWb     = wls_build_AtWb_memory(x, y, Psat, G_target, w_alpha, w_pow, t_kv, deg, K, terms);
    c_flat   = AtWA_full \ AtWb;
    c_update = reshape(c_flat, K, T_terms);
    c_matrix = c_matrix + mu * c_update;

    % (g) Per-iteration plot
    fh = figure('Visible','off','Position',[100 100 1100 700]);
    subplot(2,1,1);
    plot(abs(RefSignalX)); hold on; plot(abs(y));
    legend('|RefSignalX|','|y|'); grid on;
    title(sprintf('Mem B-spline iter %d NMSE=%.2f dB max|xd|=%.3f', k, nmse(k), max(abs(xd))));
    subplot(2,1,2);
    plot(abs(x), abs(y), '.'); hold on; plot(abs(x), abs(x), 'k-');
    grid on; xlabel('|RefSignalX|'); ylabel('|y|'); title('AM-AM');
    exportgraphics(fh, sprintf('bspline_mem_iter_%02d.png', k));
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
title('Memory B-spline DPD convergence');

figure;
plot(abs(x), abs(y0),      '.'); hold on;
plot(abs(x), abs(y_final), '.');
plot(abs(x), abs(x),       'k-');
grid on; xlabel('|RefSignalX|'); ylabel('|CompressedSignalY|');
legend('Before DPD','After DPD','Ideal','Location','NorthWest');
title('Memory B-spline DPD  AM-AM');

% Per-term LUT magnitudes (final iter) -- shows which taps got "used"
figure;
lut_final = squeeze(lut_log(:,:,end));
labels = arrayfun(@(t) sprintf('Term %d: x(n-%d)*G(|x(n-%d)|)', t, terms(t).m1, terms(t).m2), ...
                  1:T_terms, 'UniformOutput', false);
subplot(2,1,1);
plot(r_lut, 20*log10(abs(lut_final) + 1e-12), '-o'); grid on;
xlabel('|x|'); ylabel('|G_i| (dB)'); legend(labels, 'Location', 'best');
title('Memory B-spline per-tap LUT magnitudes (final iter)');
subplot(2,1,2);
plot(r_lut, unwrap(angle(lut_final)) * 180/pi, '-o'); grid on;
xlabel('|x|'); ylabel('\angle G_i (deg)'); legend(labels, 'Location', 'best');
title('Memory B-spline per-tap LUT phases');

sa1 = dsp.SpectrumAnalyzer('SampleRate',BasebandSampleRate*1e6,'SpectralAverages',5, ...
    'ShowLegend',true,'ChannelNames',{'RefSignalX','Without Memory B-spline DPD','With Memory B-spline DPD'});
sa1.YLimits = [-10,100];
sa1([RefSignalX(:), y0(:), y_final(:)]);

% Save coefficients
save('bspline_mem_coeffs.mat', 'c_matrix', 'lut_log', 'nmse', 'NMSE0', 'terms');
fprintf('\nSaved coefficients to bspline_mem_coeffs.mat\n');

%%
% =============================================================================
% Local functions
% =============================================================================

function xd = applyMemoryLUT(x, luts, terms, r_lut)
% Apply memory B-spline DPD:
%   xd(n) = sum_i  G_i(|x(n - m2_i)|) * x(n - m1_i)
% First M_max samples passed through unchanged (no memory available).
    x = x(:);
    N = length(x);
    T_terms = length(terms);
    M = max([terms.m1, terms.m2]);

    xd = zeros(N, 1);
    xd(1:M) = x(1:M);

    dr = r_lut(2) - r_lut(1);
    idx_core = (M+1):N;
    xd_core = zeros(length(idx_core), 1);

    for i_t = 1:T_terms
        m1 = terms(i_t).m1;
        m2 = terms(i_t).m2;
        x_m1 = x(idx_core - m1);
        r_m2 = abs(x(idx_core - m2));

        r_c  = min(r_m2, r_lut(end));
        t    = r_c / dr;
        i0   = min(floor(t), length(r_lut)-1);
        i_lo = i0 + 1;
        i_hi = min(i_lo + 1, length(r_lut));
        frac = t - double(i0);
        G_i  = (1 - frac) .* luts{i_t}(i_lo) + frac .* luts{i_t}(i_hi);

        xd_core = xd_core + G_i .* x_m1;
    end
    xd(idx_core) = xd_core;
end

function AtWA = wls_build_AtWA_memory(x, w_alpha, w_pow, t_kv, deg, K, terms)
% Build the (T_terms*K) x (T_terms*K) weighted normal matrix.
% Block (i, j) is the K x K matrix of cross-products between term i and term j.
    T_terms = length(terms);
    total_K = K * T_terms;
    AtWA = zeros(total_K, total_K);
    N = length(x);
    M = max([terms.m1, terms.m2]);

    for n = (M+1):N
        rv_n = abs(x(n));
        w    = 1 + w_alpha * rv_n^w_pow;
        sw   = sqrt(w);

        phi_blocks = cell(T_terms, 1);
        first_ks   = zeros(T_terms, 1);
        for i_t = 1:T_terms
            m1 = terms(i_t).m1;
            m2 = terms(i_t).m2;
            rv_m2 = abs(x(n - m2));
            [vals, fk] = bspline_eval4(rv_m2, t_kv, deg);
            phi_blocks{i_t} = (vals(:) * x(n - m1)) * sw;   % (4 x 1) complex
            first_ks(i_t)   = fk;
        end

        for i_t = 1:T_terms
            ro = (i_t - 1) * K;
            ri = ro + (first_ks(i_t) : first_ks(i_t) + 3);
            for j_t = 1:T_terms
                co = (j_t - 1) * K;
                ci = co + (first_ks(j_t) : first_ks(j_t) + 3);
                AtWA(ri, ci) = AtWA(ri, ci) + conj(phi_blocks{i_t}) * phi_blocks{j_t}.';
            end
        end
    end
end

function AtWb = wls_build_AtWb_memory(x, y, Psat, G_target, w_alpha, w_pow, t_kv, deg, K, terms)
% Build the (T_terms*K) x 1 weighted right-hand side using output-space error.
    T_terms = length(terms);
    total_K = K * T_terms;
    AtWb = zeros(total_K, 1);
    N = length(x);
    M = max([terms.m1, terms.m2]);

    for n = (M+1):N
        rv_n = abs(x(n));
        w    = 1 + w_alpha * rv_n^w_pow;
        sw   = sqrt(w);

        r_tgt = min(G_target * rv_n, Psat);
        x_tgt = r_tgt / max(rv_n, 1e-12) * x(n);
        err_sw = (x_tgt - y(n)) * sw;

        for i_t = 1:T_terms
            m1 = terms(i_t).m1;
            m2 = terms(i_t).m2;
            rv_m2 = abs(x(n - m2));
            [vals, fk] = bspline_eval4(rv_m2, t_kv, deg);
            phi_i = (vals(:) * x(n - m1)) * sw;
            rows = (i_t - 1) * K + (fk : fk + 3);
            AtWb(rows) = AtWb(rows) + conj(phi_i) * err_sw;
        end
    end
end

% --- Unchanged helpers from Bspline_DPD_PlutoSDR_2.m ---

function [CompressedSignalY, SyncOffset] = captureAndSync(s, rx, tx, RefSignalX, pad, frameSize, Cfarrow, soPrev) %#ok<INUSD>
    txWave = complex([zeros(pad,1); s(:); zeros(pad,1)]);
    release(tx);
    tx.transmitRepeat(txWave);

    for i = 1:2, rx(); end
    [d, ~, of] = rx();
    if of, fprintf('  (RX overflow)\n'); end
    dataVec = d(:);

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
    CompressedSignalY = CompressedSignalY - mean(CompressedSignalY);
    CompressedSignalY = CompressedSignalY * (rms(RefSignalX) / rms(CompressedSignalY));
    phi  = angle(CompressedSignalY' * RefSignalX);
    CompressedSignalY = CompressedSignalY * exp(1j*phi);
end

function y = farrowEval(dataVec, SyncOffset, n, mu, Cfarrow)
    di = floor(mu);
    f  = mu - di;
    m  = SyncOffset + n + di;
    X  = [dataVec(m-1), dataVec(m), dataVec(m+1), dataVec(m+2)];
    P  = X * Cfarrow.';
    y  = ((P(:,1)*f + P(:,2))*f + P(:,3))*f + P(:,4);
end

function lut = capLUT(lut, G_max, r_lut, r_drive, r_sat)
    g_lim = G_max * ones(size(lut));
    if nargin >= 4 && ~isempty(r_drive)
        g_lim = min(g_lim, r_drive ./ max(r_lut, eps));
    end
    mag  = abs(lut);
    over = mag > g_lim;
    lut(over) = lut(over) ./ mag(over) .* g_lim(over);
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

function B = buildBspl(r_nodes, t_kv, deg, K)
    r_nodes = r_nodes(:);
    B = zeros(numel(r_nodes), K);
    for ii = 1:numel(r_nodes)
        [vals, first_k] = bspline_eval4(r_nodes(ii), t_kv, deg);
        B(ii, first_k:first_k+3) = vals(:).';
    end
end

function [vals, first_k] = bspline_eval4(r_val, t_kv, deg)
    K    = length(t_kv) - deg - 1;
    nSeg = length(t_kv) - 2*deg - 1;
    r_val = max(0, min(1, r_val));
    if r_val >= 1.0
        span_0 = K - 1;
    else
        span_0 = deg + floor(r_val * nSeg);
        if span_0 > K - 1, span_0 = K - 1; end
    end
    first_k_0 = span_0 - deg;
    N0 = zeros(deg + 2, 1);
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
    vals = N0(1:4);
    first_k = first_k_0 + 1;
    if first_k < 1, first_k = 1; end
    if first_k + 3 > K, first_k = K - 3; end
end
