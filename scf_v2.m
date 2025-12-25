function [p, pgp, ASYMCORR] = scf_v2(S, H, H_interpolant, opts, data_folder)
% SCF-based correction estimator and thermalization engine

% ---------------- Initialization ----------------
opts = parse_scf_options(opts);
phys = derive_relaxation_scales(S);
geom = init_correction_geometry(S);
pdf  = init_pdf_thermalizer(S, geom);
scf  = init_scf_controller(S, geom, phys, opts);

thermflag = 0;
graphing=true;

% ---------------- File naming ----------------
if S.potential==1, potname='lj';
elseif S.potential==2, potname='wca';
else, potname='hs'; end

seriesname = 'scf';
filenamecorrection = sprintf('ASYMCORR_%s_%s_%.0e_%.0e_%.0f.mat', ...
    seriesname, potname, S.rp, S.phi, S.N);

filestart = sprintf('START_SBC_%s_%s_%.0e_%.0e_%.0f.mat', ...
    seriesname, potname, S.rp, S.phi, S.N);

% ---------------- Initial configuration ----------------
disp('Creating initial configuration...');
p = startingPositions_lj(S);
disp('MC shaker to erase memory...');
for k = 1:1e5
    idx = randi(S.N);
    trial = p(idx,:) + (rand(1,3)-0.5)*S.br;
    if norm(trial) < S.br
        d2 = pdist2(trial, p, 'squaredeuclidean');
        d2(idx) = inf;
        if min(d2) > (2*S.rp)^2
            p(idx,:) = trial;
        end
    end
end

pgp = p - (2*S.br) * (p ./ (vecnorm(p,2,2)+eps));

% ---------------- Noise library ----------------
disp('Building noise library...');
DISP = build_noise_library(S.stdx, 1e6);
qd = 1;
qs = 0;

% ===================== MAIN LOOP =====================
persistent ndens_acc

if isempty(ndens_acc)
    ndens_acc.edges   = linspace(0,S.br,100);
    ndens_acc.centers = ndens_acc.edges(1:end-1) + diff(ndens_acc.edges)/2;
    ndens_acc.vols    = (4/3)*pi*(ndens_acc.edges(2:end).^3 ...
                                - ndens_acc.edges(1:end-1).^3);
    ndens_acc.counts  = zeros(numel(ndens_acc.centers),1);
    ndens_acc.nsamples = 0;
end

while true
    qs = qs + 1;

    prho  = vecnorm(p,2,2);
    pvers = p ./ (prho + eps);
    idxgp = prho > (S.br - S.rc);

    pairdists = pdist(p);

    % -------- Thermalization --------
    if thermflag == 0
		if qs==1
            disp('Thermalizing...');
        end
        [pdf, done] = update_pdf_thermalizer(pdf, pairdists, qs, opts, phys);
        if done
            disp('--- Thermalization Complete ---');
            thermflag = 1;
            pdf.nsamples=qs;
            qs = 0;

            scf.acc.sum(:) = 0;
            scf.acc.sum_sq(:) = 0;
            scf.acc.counts(:) = 0;

            pdf.counts(:) = 0;
            pdf.pass_count = 0;
            pdf.prev_rms = inf;
            continue
        end
    end

    % -------- Displacements --------
    v = DISP(qd:qd+S.N-1,:);
    vgp = DISP(qd+S.N:qd+2*S.N-1,:);
    qd = qd + 2*S.N;
    if qd+2*S.N > 1e6
        DISP = DISP(randperm(1e6),:);
        qd = 1;
    end

    if S.potential ~= 0
        ptemp = [p; pgp(idxgp,:)];
        dpot = potential_displacements_v13(ptemp, S, H, H_interpolant, 1);
        v = v + dpot(1:S.N,:);
        vgp(idxgp,:) = vgp(idxgp,:) + dpot(S.N+1:end,:);
    end

    % -------- SCF correction --------
    if thermflag == 1
        pdf.nsamples=pdf.nsamples+1;
        scf = scf_update_controller(scf, prho, pvers, v, qs);
    end

    % -------- Apply correction --------
    drc = interp1(scf.centers, scf.correction, prho, 'linear', 0);
    drc(prho < (S.br - geom.depth)) = 0;

    p2 = p + v + drc .* pvers;
    p2rho = vecnorm(p2,2,2);

    % -------- Ghosts --------
    pgp_dir = pgp ./ (vecnorm(pgp,2,2)+eps);
    v_rad = sum(vgp .* pgp_dir,2);
    v_tan = vgp - v_rad .* pgp_dir;

    pgp_tmp = pgp + v_tan;
    pgp_dir2 = pgp_tmp ./ (vecnorm(pgp_tmp,2,2)+eps);
    pgp2 = pgp_dir2 .* (2*S.br - p2rho);

    % -------- Promotion / demotion --------
    swap = p2rho > vecnorm(pgp2,2,2);
    p(swap,:) = pgp2(swap,:);
    pgp(swap,:) = p2(swap,:);
    p(~swap,:) = p2(~swap,:);
    pgp(~swap,:) = pgp2(~swap,:);

    % ----------- Density accumulation ----------
    ndens_acc.counts  = ndens_acc.counts + histcounts(p2rho, ndens_acc.edges)';
    ndens_acc.nsamples = ndens_acc.nsamples + 1;


    if graphing && mod(qs, 5000) == 0
        state = struct( ...
            'p',p, ...
            'pgp',pgp, ...
            'S',S, ...
            'pdf',pdf, ...
            'scf',scf, ...
            'step',qs, ...
            'ndens_acc',ndens_acc);
        plot_scf_diagnostics(state);
        drawnow
    end


    if qs > 1e6
        warning('Safety stop reached');
        break
    end
end

% ---------------- Outputs ----------------
ASYMCORR.correction = [scf.centers, scf.correction];
ASYMCORR.history    = [];
ASYMCORR.sgd_edges  = scf.edges;
ASYMCORR.S = S;
ASYMCORR.opts = opts;

if opts.io.enable
    save(fullfile(data_folder, filenamecorrection), 'ASYMCORR');
    save(fullfile(data_folder, filestart), 'p', 'pgp', 'S');
end

disp('SCF complete.');
end



function opts = parse_scf_options(opts_in)

    % ---------------- IO ----------------
    opts.io.enable   = get_opt(opts_in, 'enable_io', true);
    opts.io.graphing = get_opt(opts_in, 'graphing',  true);
    opts.io.debug    = get_opt(opts_in, 'debugging', false);

    % ---------------- SCF CONTROL ----------------
    opts.scf.update_interval = get_opt(opts_in, 'update_interval', 10000);
    opts.scf.max_gain        = get_opt(opts_in, 'max_gain',        0.10);
    opts.scf.gain_prefactor  = get_opt(opts_in, 'gain_prefactor',  0.02);
    opts.scf.smooth_width    = get_opt(opts_in, 'smooth_width',    15);
    opts.scf.memory_floor    = get_opt(opts_in, 'memory_floor',     0.5);

    % ---------------- THERMALIZATION ----------------
    opts.thermal.required_passes = get_opt(opts_in, 'therm_passes', 100);
    opts.thermal.rms_limit       = get_opt(opts_in, 'therm_rms',    0.2);

end

% -------- helper ----------
function v = get_opt(s, field, default)
    if isstruct(s) && isfield(s, field)
        v = s.(field);
    else
        v = default;
    end
end

function phys = derive_relaxation_scales(S)

    gCS = (1 - S.phi/2) / (1 - S.phi)^3;
    diffE = S.esdiff * S.alpha / gCS;

    tau_alpha = (S.rp^2) / (6 * diffE);

    phys.diffusivity         = diffE;
    phys.relax_steps         = ceil(tau_alpha / S.timestep);
    phys.decorrelation_steps = ceil(5 * phys.relax_steps);

end

function geom = init_correction_geometry(S)

    taper_width = 1.0 * S.rp;
    potdepth    = min(S.br, S.rc + taper_width);

    geom.depth  = potdepth;
    geom.edges  = sort((S.br:-0.05*S.rp:S.br - potdepth)');
    geom.centers = geom.edges(1:end-1) + diff(geom.edges)/2;

    r0 = min(geom.edges);
    geom.taper = min(1, max(0, (geom.centers - r0) / taper_width));

end

function pdf = init_pdf_thermalizer(S, geom)

    maxdist = 2 * S.br;
    dr = 0.05 * S.rp;

    edges = sort((maxdist:-dr:0)');
    centers = edges(1:end-1) + diff(edges)/2;

    vols = (4/3)*pi*(edges(2:end).^3 - edges(1:end-1).^3);

    ndens0 = S.N / S.bv;
    rnorm  = centers / S.br;

    geom_factor = max(0, 1 - (3/4)*rnorm + (1/16)*rnorm.^3);
    denom = 0.5 * (ndens0 * geom_factor * S.N) .* vols;

    mask = centers > 2*(S.br - geom.depth) & ...
           centers < 2*S.br - 2*S.rp;

    pdf.edges  = edges;
    pdf.centers = centers;
    pdf.denom  = denom;
    pdf.mask   = mask;

    pdf.counts     = zeros(numel(centers),1);
    pdf.prev_rms   = inf;
    pdf.pass_count = 0;

end

function scf = init_scf_controller(S, geom, phys, opts)

    scf.edges   = geom.edges;
    scf.centers = geom.centers;
    scf.taper   = geom.taper;

    nb = numel(scf.centers);

    scf.correction = zeros(nb,1);

    scf.acc.sum    = zeros(nb,1);
    scf.acc.sum_sq = zeros(nb,1);
    scf.acc.counts = zeros(nb,1);

    scf.update_interval = opts.scf.update_interval;
    scf.max_gain        = opts.scf.max_gain;
    scf.gain_prefactor  = opts.scf.gain_prefactor;
    scf.smooth_width    = opts.scf.smooth_width;
    scf.memory_floor   = opts.scf.memory_floor;

end

function [scf, did_update] = scf_update_controller( ...
        scf, prho, pvers, base_disp, step_index)

    did_update = false;

    % ------------------------------------------------------------
    % 1. Accumulate radial drift statistics (every step)
    % ------------------------------------------------------------
    dr = sum(base_disp .* pvers, 2);      % radial displacement
    bin = discretize(prho, scf.edges);

    valid = bin > 0 & bin <= numel(scf.centers);

    if any(valid)
        idx = bin(valid);
        val = dr(valid);

        scf.acc.sum     = scf.acc.sum     + accumarray(idx, val,   size(scf.acc.sum));
        scf.acc.sum_sq  = scf.acc.sum_sq  + accumarray(idx, val.^2,size(scf.acc.sum));
        scf.acc.counts = scf.acc.counts + accumarray(idx, 1,     size(scf.acc.sum));
    end

    % ------------------------------------------------------------
    % 2. Apply correction only every update_interval
    % ------------------------------------------------------------
    if mod(step_index, scf.update_interval) ~= 0
        return
    end

    counts = scf.acc.counts;
    active = counts > 1;

    if ~any(active)
        return
    end

    % ------------------------------------------------------------
    % 3. Estimate mean drift and statistical uncertainty
    % ------------------------------------------------------------
    mu  = zeros(size(counts));
    sem = inf(size(counts));

    mu(active) = scf.acc.sum(active) ./ counts(active);

    var_bin = (scf.acc.sum_sq(active) ...
              - counts(active).*mu(active).^2) ...
              ./ max(1, counts(active)-1);

    sem(active) = sqrt(max(0, var_bin)) ./ sqrt(counts(active));

    % ------------------------------------------------------------
    % 4. Signal-to-noise gated gain
    % ------------------------------------------------------------
    snr = abs(mu) ./ (sem + eps);

    gain = scf.gain_prefactor * max(0, snr - 1);
    gain = min(gain, scf.max_gain);

    update = gain .* mu;

    if ~any(update ~= 0)
        return
    end

    % ------------------------------------------------------------
    % 5. Apply tapered, smoothed correction
    % ------------------------------------------------------------
    scf.correction = scf.correction - update .* scf.taper;
    scf.correction = smoothdata(scf.correction, 'movmean', scf.smooth_width);

    % ------------------------------------------------------------
    % 6. Memory-weighted forgetting
    % ------------------------------------------------------------
    forget = 1 - min(1, gain * 5);
    forget = max(scf.memory_floor, forget);

    scf.acc.sum     = scf.acc.sum     .* forget;
    scf.acc.sum_sq  = scf.acc.sum_sq  .* forget;
    scf.acc.counts = scf.acc.counts .* forget;

    did_update = true;

end

function [pdf, is_thermalized] = update_pdf_thermalizer( ...
        pdf, pairdists, step_index, opts, phys)

    is_thermalized = false;

    % ------------------------------------------------------------
    % 1. Accumulate pair-distance histogram (every step)
    % ------------------------------------------------------------
    hc = histcounts(pairdists, pdf.edges);

    if numel(hc) == numel(pdf.counts)
        pdf.counts = pdf.counts + hc';
    end

    % ------------------------------------------------------------
    % 2. Only evaluate every thermal block
    % ------------------------------------------------------------
    if step_index < phys.relax_steps
        return
    end

    % ------------------------------------------------------------
    % 3. Compute cumulative g(r)
    % ------------------------------------------------------------
    g = (pdf.counts / step_index) ./ pdf.denom;

    residuals = g(pdf.mask) - 1;
    rms = sqrt(mean(residuals.^2));

    % ------------------------------------------------------------
    % 4. Estimate statistical noise floor
    % ------------------------------------------------------------
    expected = pdf.denom(pdf.mask) * step_index;
    expected(expected == 0) = inf;

    sigma = sqrt(mean(1 ./ expected));

    % ------------------------------------------------------------
    % 5. Convergence criteria
    % ------------------------------------------------------------
    drift_ok = abs(rms - pdf.prev_rms) < 5 * (sigma / phys.relax_steps);
    rms_ok   = rms < opts.thermal.rms_limit;

    if drift_ok && rms_ok
        pdf.pass_count = pdf.pass_count + 1;
    else
        pdf.pass_count = 0;
    end

    pdf.prev_rms = rms;

    % ------------------------------------------------------------
    % 6. Declare thermalization complete
    % ------------------------------------------------------------
    if pdf.pass_count >= opts.thermal.required_passes
        is_thermalized = true;
    end

end

function [fig, ax] = init_scf_figure()

    fig = figure( ...
        'Color','k', ...
        'Name','SBC / SCF Diagnostics', ...
        'Units','normalized', ...
        'Position',[0.05 0.05 0.9 0.85]);
    
    t = tiledlayout(fig,2,2,'TileSpacing','compact','Padding','compact');
    
    ax.ndens = nexttile(t);
    ax.pdf   = nexttile(t);
    ax.scf   = nexttile(t);
    ax.snr   = nexttile(t);
    
    set(struct2array(ax),'Color','k','XColor','w','YColor','w')

end




function plot_scf_diagnostics(state)

S   = state.S;
p   = state.p;
pgp = state.pgp;
pdf = state.pdf;
scf = state.scf;
qs  = state.step;
ndens_acc  = state.ndens_acc;

persistent fig ax
if isempty(fig) || ~isvalid(fig)
    [fig, ax] = init_scf_figure();
end

% ------------------------------------------------------------
% 1. RADIAL NUMBER DENSITY (REAL + GHOST)
% ------------------------------------------------------------
ndens = ndens_acc.counts ./ (ndens_acc.nsamples * ndens_acc.vols);
ndens = ndens / mean(ndens(ndens_acc.centers < 0.6*S.br));


cla(ax.ndens)
plot(ax.ndens, ndens_acc.centers/S.br, ndens,'w','LineWidth',1.8)
yline(ax.ndens,1,'--','Color',[0.6 0.6 0.6])
xlabel(ax.ndens,'r / R')
ylabel(ax.ndens,'\rho(r)/\rho_0')
title(ax.ndens,'Radial Number Density')
ylim(ax.ndens,[0.8 1.2])
grid(ax.ndens,'on')

% ------------------------------------------------------------
% 2. PDF DEVIATION NEAR BOUNDARY
% ------------------------------------------------------------
g = (pdf.counts / pdf.nsamples) ./ pdf.denom;

cla(ax.pdf)
plot(ax.pdf, pdf.centers/S.br, g-1,'c','LineWidth',1.5)
yline(ax.pdf,0,'--','Color',[0.5 0.5 0.5])
xlabel(ax.pdf,'r / R')
ylabel(ax.pdf,'g(r) - 1')
title(ax.pdf,'Pair Correlation Deviation')
xlim(ax.pdf,[1.2 2.0])
ylim(ax.pdf,[-0.3 0.3])
grid(ax.pdf,'on')

% ------------------------------------------------------------
% 3. SCF CORRECTION PROFILE
% ------------------------------------------------------------
cla(ax.scf)
plot(ax.scf, scf.centers/S.br, scf.correction,'m','LineWidth',2)
yline(ax.scf,0,'--','Color',[0.6 0.6 0.6])
xlabel(ax.scf,'r / R')
ylabel(ax.scf,'\Delta F_r')
title(ax.scf,'SCF Radial Correction')
grid(ax.scf,'on')

% ------------------------------------------------------------
% 4. SCF SIGNAL-TO-NOISE
% ------------------------------------------------------------
counts = scf.acc.counts;
mu = zeros(size(counts));
sem = inf(size(counts));

active = counts > 1;
mu(active) = scf.acc.sum(active) ./ counts(active);
var_bin = (scf.acc.sum_sq(active) - counts(active).*mu(active).^2) ...
           ./ max(1,counts(active)-1);
sem(active) = sqrt(max(0,var_bin)) ./ sqrt(counts(active));
snr = abs(mu)./(sem+eps);

cla(ax.snr)
semilogy(ax.snr, scf.centers/S.br, snr,'y','LineWidth',1.5)
yline(ax.snr,1,'--','SNR=1','Color',[0.7 0.7 0.7])
xlabel(ax.snr,'r / R')
ylabel(ax.snr,'|μ| / σ')
title(ax.snr,'SCF Signal-to-Noise')
grid(ax.snr,'on')

% ------------------------------------------------------------
% 5. TITLE + TIMESTAMP
% ------------------------------------------------------------
sgtitle(fig,sprintf('SCF Diagnostics — step %d',qs),'Color','w')

drawnow limitrate
end

