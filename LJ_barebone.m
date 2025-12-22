%% CLEANUP

clear all
close all
rng('shuffle')
warning('off', 'all');
% debugging=true

%% ADD PATHS TO STORAGE FOLDER

if exist('D:\GoogleDrive\LCL\Ludovico Cademartiri\Work\projects\ARBD\database','dir')
    data_folder = 'D:\GoogleDrive\LCL\Ludovico Cademartiri\Work\projects\ARBD\database';
elseif exist('G:\My Drive\LCL\Ludovico Cademartiri\Work\projects\ARBD\database','dir')
    data_folder = 'G:\My Drive\LCL\Ludovico Cademartiri\Work\projects\ARBD\database';
elseif exist('D:\GDrive\LCL\Ludovico Cademartiri\Work\projects\ARBD\database','dir')
    data_folder = 'D:\GDrive\LCL\Ludovico Cademartiri\Work\projects\ARBD\database';
end
if exist('D:\OneDrive - Università degli Studi di Parma\Manuel Dedola\outlines\spherical boundary conditions for brownian dynamics simulations\submissions\JChemPhys\revision\code\freeghost_nocorr','dir')
    output_folder='D:\OneDrive - Università degli Studi di Parma\Manuel Dedola\outlines\spherical boundary conditions for brownian dynamics simulations\submissions\JChemPhys\revision\code\freeghost_nocorr';
elseif exist('C:\Users\lcade\OneDrive - Università degli Studi di Parma\Manuel Dedola\outlines\spherical boundary conditions for brownian dynamics simulations\submissions\JChemPhys\revision\code\freeghost_nocorr','dir')
    output_folder='C:\Users\lcade\OneDrive - Università degli Studi di Parma\Manuel Dedola\outlines\spherical boundary conditions for brownian dynamics simulations\submissions\JChemPhys\revision\code\freeghost_nocorr';
elseif exist('D:\OneDrive - Università degli Studi di Parma\Manuel Dedola\outlines\spherical boundary conditions for brownian dynamics simulations\submissions\JChemPhys\revision\code\freeghost_nocorr','dir')
    output_folder='D:\OneDrive - Università degli Studi di Parma\Manuel Dedola\outlines\spherical boundary conditions for brownian dynamics simulations\submissions\JChemPhys\revision\code\freeghost_nocorr';
end
    
toolbox_folder = '..\ARBD_toolbox';
addpath(data_folder)
addpath(toolbox_folder)
addpath(output_folder)

%% SERIES NAME

filenamecollisionseries='testing_%d_%d.mat';
filenamecollisionseries_temp='testing_temp_%d_%d.mat';

%% FIXED PHYSICAL PARAMETERS

C.T=298.15; %K (temperatura)
C.kb=1.38e-23; %J*K^-1 (costante di Boltzmann)
C.kbT=C.kb*C.T;
C.epsilon=1*C.kbT;
C.hydrationlayer=2.5e-10;

%% OPTIONAL SWITCHES

P.allpos=1;
P.ghostghost=1;
P.io_enabled=true;
P.correctionwindow=0; % window of the correction; 0 = no correction, 1e6 = S.rc, any other value X=X*S.rp;
P.convergencemode=1; % 1  by step numbers, 2 by coll numbers, 3 by coll rates convergence
P.pdfbins=[180,90]; % number of bins for azimuth, elevation
P.pdfthermints=10; % interval of steps between snapshots of pdfs to ascertain thermalization
P.rfstepsforeffdiffest=10000; % number of random walk steps that I need to collect to extract the alpha parameter

%% SIMULATION PARAMETERS

P.nodisp=1e7;  % size of displacement library
P.reps=1; % number of replicates
P.maxcoll=1e7; % minimum number of collisions to measure
P.maxsteps=1e6; % maximum number of steps to make when doing pdfs
P.kuhnmultiplier=200; % multiplier of taur that is used to either include or exclude persistence
P.kuhnmultiplierVACF=200; % multiplier of taur that is used to calculate autocorrelation
P.convergencewindows=[10000,1000]; % number of timesteps over which to describe the evolution of k in time; number of steps over which to evaluate convergence
P.evinterval=1e4; % ncoll interval between excluded volume estimations
P.noevpoints=1e5; % number of points generated to estimate accessible volume
P.ratio=1000; % maximum number of physical trajectory steps in each relaxation time to calculate the effective diffusivity. This is the number that ensures that the timescale is adaptive

%% VARIABLE SPACE and ASSUMPTIONS

C.solventproperties=solventlibrary();
C.phasesproperties=phaseslibrary();

V.solvent_list=[1]; % list of solvents according to the library above
V.phase_list=[2]; % list of phases according to the library above
V.r_range=[1e-08,1e-8,1,1]; % min, max, steps of particle radii, then log switch (1 if logscale)
V.N_range=[1e3,1e3,1,1]; % min, max, steps of particle numbers, then log switch (1 if logscale)
V.phi_range=[1e-3,0.40,5,1]; % min, max, steps of volume fractions, then log switch (1 if logscale)
V.bc_range=[1,1,1]; % min, max, steps of boundary conditions (1:SBC; 2:PBC_cubic; 3:PBC_fcc; 4:BB)
V.bbm_range=[1,1,1]; % big box multiplier
V.LJpot_range=[1,1,1]; % LJ potential trigger (0 = HS; 1 = LJ; 2 = WCA)
V.T_range=[298.15,298.15,1]; % min, max, steps of temperatures

V.assumptions(1)=1; % exclude those where the Kuhn time is smaller than the trajectory step time
V.assumptions(2)=1; % exclude those where the particle is smaller than the solvent
V.assumptions(3)=1; % exclude those where the particle is lighter than the solvent

[V,CONDS]=simconditions_LJ(V,C,P);

%% EFFECTIVE DIFFUSIVITY CALCULATION

disp('collection and estimation of effective diffusivities for the conditions to be analyzed')

CONDS=effective_diffusivity(data_folder,CONDS,P,C);

%% SIMULATION EXECUTION

for ic=5

    if CONDS.alpha(ic,1)==0
        continue
    end

    % --- initialize main variables from CONDS struct array
    S.timestep=V.c{ic,11};
    S.kbT=C.kbT;
    S.bc=CONDS.bc(ic,1);
    S.br=CONDS.br(ic,1);
    S.bv=CONDS.bv(ic,1);    
    S.rp=CONDS.rp(ic,1);
    S.N=CONDS.N(ic,1);
    S.phi=CONDS.phi(ic,1);
    S.solvent=CONDS.solvent(ic,1);
    S.phase=CONDS.phase(ic,1);
    S.alpha=CONDS.alpha(ic,1);
    S.esdiff=C.kbT/(6*pi*CONDS.eta(ic,1)*S.rp);
    S.stdx=sqrt(2*S.alpha*S.esdiff*CONDS.kt(ic,1));
    S.kt=CONDS.kt(ic,1);
    S.bbm=CONDS.bbm(ic,1);
    S.potential=CONDS.pot(ic);
    S.pot_corr=true;
    if S.potential>0
        S.pot_epsilon=C.epsilon;
        S.pot_sigma=2*S.rp+C.hydrationlayer;
        S.rc=3*S.pot_sigma;
        P.pdfsigmaplusthresh=2.76; % particle radius multipler within which to collect PDFsigma data
    else
        S.rc=2*S.rp;
        P.pdfsigmaplusthresh=3.2; % particle radius multipler within which to collect PDFsigma data
    end
    if P.correctionwindow==0
        S.correctionwindow=0;
    elseif P.correctionwindow==1e6
        S.correctionwindow=S.rc;
    else
        S.correctionwindow=P.correctionwindow*S.rp;
    end
    [~, cmdout] = system('wmic cpu get L2CacheSize, L3CacheSize /value');
    tokens = regexp(cmdout, '\d+', 'match');
    S.cacheSizeMB = (max(str2double(tokens))/1024)/feature('numCores');
    S.gtrig=S.rc;
    if S.bc==3 % THIS IF CONDITION IS VERIFIED AND CORRECT
        S.fcc.unitvecs=V.fcc_unitvecs;
        S.fcc.a1 = 2*S.br.*S.fcc.unitvecs(1,:);
        S.fcc.a2 = 2*S.br.*S.fcc.unitvecs(2,:);
        S.fcc.a3 = 2*S.br.*S.fcc.unitvecs(3,:);
        S.fcc.A=[S.fcc.a1(:), S.fcc.a2(:), S.fcc.a3(:)]';
        S.fcc.invA = inv(S.fcc.A);
        S.fcc.reciprocal_vectors = S.fcc.invA';
        S.fcc.normals = [cross(S.fcc.a2,S.fcc.a3);
           cross(S.fcc.a3,S.fcc.a1);
           cross(S.fcc.a1,S.fcc.a2)];
        S.fcc.normals = S.fcc.normals ./ vecnorm(S.fcc.normals,2,2);
        S.fcc.frac_thresh = S.rc ./ abs(sum(S.fcc.A'.*S.fcc.normals,2))';
        [n1, n2, n3] = ndgrid(-1:1);
        S.fcc.shift_coeffs = [n1(:), n2(:), n3(:)];
        S.fcc.shift_coeffs(all(S.fcc.shift_coeffs==0,2),:) = []; % A robust way to remove [0,0,0]
        S.fcc.cart_shifts = S.fcc.shift_coeffs * S.fcc.A;
        clear n1 n2 n3
    elseif S.bc==2 || S.bc==1
        S.fcc.A = diag([2*S.br,2*S.br,2*S.br]);
        S.fcc.invA = diag([1/(2*S.br),1/(2*S.br),1/(2*S.br)]); 
    end
    clear nx ny nz dx dy dz SX SY SZ neighbor_offsets neighbor_linear valid lin all_ids remainder cx cy cz
    
    IDlist=double(linspace(1,S.N,S.N)'); % id list for all particles in the boundary to use when the position matrix 'p' is rebuilt at the beginning of every time step
    % ---

    % --- CALCULATION OF LJ FORCE -----
    if S.potential~=0
        clear H
        H=pot_force(S.potential,S.rc,30000,S.pot_sigma,S.pot_epsilon);
        H_interpolant = griddedInterpolant(H(:,1), H(:,2), 'linear', 'nearest');
    end
    % ---

    % --- loop over replicates
    for irep=1:P.reps % loop over replicates 

        % --- filename    
        filename=sprintf(filenamecollisionseries,ic,irep);
        tempfilename=sprintf(filenamecollisionseries_temp,ic,irep);
        m = matfile([output_folder,'\',tempfilename], 'Writable', true);
        %---

        % --- continue if filename exists
        if exist(filename,'file')>0 
           % continue
        end
        % ---

        % --- chunking
        chunk_size = 1e4;
        num_chunks = ceil(P.maxsteps / chunk_size);
        if S.bc==1
            m.POS = zeros(2*S.N, 3, P.maxsteps, 'single');
            pos_buffer = zeros(2*S.N, 3, chunk_size, 'single');
        else
            m.POS = zeros(S.N, 3, P.maxsteps, 'single');
            pos_buffer = zeros(S.N, 3, chunk_size, 'single');
        end
        m.S = S;
        m.V = V;
        m.P = P;
        m.C = C;
        m.ic = ic;
        buffer_idx = 1;
        % ---

        % ---- CALCULATION OF DISPLACEMENT LIBRARIES ----
        DISP=build_noise_library(S.stdx,P.nodisp);
        % ----

        % --- MONTECARLO CLAMP DETERMINATION ---
        if S.potential~=0
            S.pot_clamp=mcdClamp(P.nodisp,DISP,S.esdiff,S.timestep,H(H(:,1) >= 0.8 * S.pot_sigma, :),C.kbT);            
        end
        % ---

        % --- STARTING POSITIONS & DETERMINING SBC RADIAL DISPLACEMENT ASYMMETRY CORRECTION ---
        if S.bc==1
            if S.potential==1, potname='lj'; elseif S.potential==2, potname='wca'; else potname='hs'; end
            filestartingconfiguration = sprintf('START_SBC_%s_%.0e_%.0e_%.0f_%.1f_%.1e.mat',...
                        potname,S.rp,S.phi,S.N,S.pot_epsilon/S.kbT,S.pot_sigma);
            if exist(filestartingconfiguration,'file')
                load(filestartingconfiguration,'p','pgp')
            else
                PDF=pdf_initialization(S,P,data_folder);
                [p,pgp]=sbc_setup_sgd_v9(S,PDF,[],data_folder);
            end
            disp('initial particle configuration - determined')
            if S.pot_corr,disp('radial displacement asymmetry - determined'); end
            % --- CREATE MASTER GHOSTLIST ----
            pgp0= struct('p',{},'ID',{},'active',{},'lastStep',{});
            pgp0(1).p=pgp;
            pgp0(1).ID=IDlist;
            pgp0(1).active=zeros(S.N,1); % ACTIVE flag
            pgp0(1).lastStep=zeros(S.N,1);% LAST STEP ACTIVE
            disp('master ghost list - created')
            % ---
        else
            p=startingPositions_lj(S);
        end
        % ---

        % ---- INITIALIZE SERVICE VARIABLES AND COUNTERS ----
        disp('initializing service variables and counters')
        qd=1; % displacement counter
        qs=1; % step counter
        kconverged=0;% flag to identify when the value of k has converged and when, therefore the sim can be concluded
        qedges=1;
        
        % ---- RUN UNTIL CONVERGENCE -----        
        disp('starting simulation')
        while kconverged==0
            
            % --- SBC GHOST ACTIVATION AND DEACTIVATION -----
            if S.bc==1
                prho=vecnorm(p,2,2); % calculate distances from origin of reals
                idxgp=prho>(S.br-S.rc); % mask of active ghosts
                pgp=pgp0.p(idxgp,:); % active ghosts positions
                ptemp=[p;pgp]; % concatenated list of reals and active ghosts
                S.Na=size(ptemp,1); % number of particles (reals + active ghosts)
            end
            % ---
            
            % ---- CALCULATION OF DISPLACEMENTS BY POTENTIALS --------------
            if S.potential~=0
                if S.bc==1 % SBC
                    % calculate the displacement components due to potentials for all particles (reals and active ghosts)
                    disppot=potential_displacements_v13(ptemp, S, H, H_interpolant, P.ghostghost);
                    % extract potential displacements for active ghosts
                    disppotgp=disppot(S.N+1:end,:);
                    % extract potential displacements for reals
                    disppot=disppot(1:S.N,:);
                else % NON-SBC conditions (PBC)
                    % Under MIC we evaluate pairwise displacements using
                    % minimum-image convention on the real particle list.
                    disppot=potential_displacements_v13(p, S, H, H_interpolant, 0,S.cacheSizeMB);
                end
            end
            % --------------------------------------------------------------

            % ---- DISPLACE PARTICLE POSITIONS -----------------------------
            if S.bc~=1 % non-SBC conditions
                displacements=DISP(qd:qd+S.N-1,:); % extract displacements for reals
                qd=qd+S.N; % update displacement library counter
                if S.potential~=0 % if not HS add potential displacement component
                    displacements=displacements+disppot;
                end
                p=[p,IDlist,displacements,p+displacements];
                % ----  BIG BOX BOUNDARY INTERACTIONS -----
                if S.bc==4
                    % this function teleports particle that touch the hard
                    % boundary to avoid progressive condensation and therefore
                    % depletion from the core of the domain
                    p=bigboxTeleport(S,p); 
                end
            elseif S.bc==1 % SBC
                % extract brownian displacements for reals and ghosts
                    displacements=DISP(qd:qd+S.Na-1,:); % extract displacements for reals and ghosts
                    qd=qd+S.Na; % update displacement library counter by reals + active ghosts
                    dispgp=displacements(S.N+1:end,:); % extract brownian displacement for active ghosts
                    displacements=displacements(1:S.N,:); % extract brownian displacement for reals
                % add potential displacements, if necessary, to both reals and ghosts displacements
                    if S.potential~=0 
                        displacements=displacements+disppot;
                        dispgp=dispgp+disppotgp;
                    end
                % convert ghost displacements to tangential only
                    pgp_norm = vecnorm(pgp,2,2);
                    pgp_norm(pgp_norm==0) = eps;
                    pgp_dir = pgp ./ pgp_norm;
                    v_rad_component = sum(dispgp .* pgp_dir, 2); % extract radial component to the total ghost displacement          
                    v_tan = dispgp - (v_rad_component .* pgp_dir); % subtract radial component from the 3D displacement
                % apply displacements to active ghosts 
                    pgp2_temp = pgp + v_tan; % move ghosts tangentially creating tentative position
                % correct real displacements by ASYMCORR
                    if S.pot_corr
                        dispcorr=zeros(size(p,1),3);
                        % mask particles within 2*S.rp of the boundary
                        idxasym = prho > (S.br - S.correctionwindow);
                        if any(idxasym)
                            % interpolate correction value for each masked particle
                            delta_r = interp1(ASYMCORR(:,1), ASYMCORR(:,2), prho(idxasym), 'pchip', 0);
                            % compute unit radial vectors
                            rho_hat = p(idxasym,:) ./ prho(idxasym);
                            % apply correction only to the radial component of the displacement
                            dispcorr(idxasym,:) = dispcorr(idxasym,:) + delta_r .* rho_hat;
                        end
                        displacements=displacements+dispcorr;
                    end
                    p2=p+displacements; % move reals creating tentative position
                % reset tether distance of initial ghosts
                    p2rho = vecnorm(p2,2,2); % get tentative real position norms
                    pgp2_temp_norm = vecnorm(pgp2_temp,2,2); % get tentative ghost positions norms
                    pgp2_temp_norm(pgp2_temp_norm==0) = eps; % (guard for tiny norms)
                    pgp2_dir = pgp2_temp ./ pgp2_temp_norm; % get versors of tentative ghost positions
                    pgp2 = pgp2_dir .* (2*S.br-p2rho(idxgp,:)); % set distance of tentative ghosts to the tether distance from tentative real positions
                    pgp2disps = pgp2-pgp; % store displacements for eventual reset of positions in case of HS potential
                    pgp=[pgp,IDlist(idxgp,:),pgp2disps,pgp2];
                % update ghost master list
                    pgp0.active(:,:)=0;
                    pgp0.active(pgp(:,4),1)=1;
                    pgp0.lastStep(pgp(:,4),:)=qs;
                    p=[p,IDlist,displacements,p+displacements]; % the p matrix contains [positions at the beginning of step, particle ID, displacements, positions at the end of the step]
                    p=[p;pgp];
            end
            % --------------------------------------------------------------

            % --- PBC POSITION WRAPPING
            % Wrap positions by MIC to canonical cell (if PBC)
            if S.bc==2 || S.bc==3
                p(:,8:10)=mic_wrap_positions(p(:,8:10), S);
            end

            % --- PROMOTION/DEMOTION IN SBC - reset in BB/PBC -------------
            if S.bc==1
                pgp0.p(pgp(:,4),:)=pgp(:,8:10);
                p=p(1:S.N,8:10);
                pgp=pgp(:,8:10);
                idxrgswap=vecnorm(p,2,2)>S.br;
                % initialize start position array for the next timestep 
                realtoswap=p(idxrgswap,:);
                ghoststoswap=pgp0.p(idxrgswap,:);                
                pgp0.p(idxrgswap,:)=realtoswap;
                p(idxrgswap,:)=ghoststoswap;
            else
                p=p(1:S.N,8:10);
            end
            % -----------------------------------------------------------

            % ----- RESHUFFLE DISPLACEMENT LIBRARIES if they run out----
            if qd+2*S.N>P.nodisp
                qd=1;
                DISP=DISP(randperm(P.nodisp),:);
            end
            % -----

            % --- CONVERGENCE CHECK AFTER MINIMUM # OF COLLISIONS IS ACHIEVED ---
            if P.convergencemode==3 && qc>P.maxcoll && mod(qs,10^4)==0
                temp=DATA{ic,1}; % first two columns are step number and number of valid 2-body colls
                temp(:,3)=temp(:,1).*S.kt; % time corresponding to the step
                temp(:,4)=cumsum(temp(:,2)); % cumulative sum of collisions
                temp(:,5)=temp(:,4)./(temp(:,3).*S.bv); % collision rate: total number of collisions divided by time and volume
                tempconc=S.N/S.bv; % particle concentration;
                temp(:,6)=temp(:,5)./(tempconc^2); % rate constant (assuming second order): collision rate/concentration^2
                % tempint is an interpolation: time and rate constant are
                % the sample points; the queried points are 10000 equispaced 
                % points between the minimum time and the maximum time.
                % This is done to have an equispaced function of
                % rateconstant(time)
                tempint=interp1(temp(:,3),temp(:,6),min(temp(:,3)):(max(temp(:,3))-min(temp(:,3)))/(P.convergencewindows(1)-1):max(temp(:,3)))';
                tempint=tempint(end-P.convergencewindows(2)+1:end,:); % extract only the last P.convwindow points (usually 1000)                
                % the convergence check is by comparing the beginning
                % values of the window with the end values and see whether
                % they have changed by more than 1%. If not the series is
                % considered converged.
                beginning=mean(tempint(1:100)); 
                ending=mean(tempint(end-100:end));
                disp((abs(ending-beginning)/beginning)*100)
                if (abs(ending-beginning)/beginning)<0.01
                    kconverged=1;
                end
            elseif P.convergencemode==1
                if qs>=P.maxsteps
                    kconverged=1;
                end
            elseif P.convergencemode==2
                if qc>=P.maxcoll
                    kconverged=1;
                end
            end
            % ---

            % --- POSITION DATA STORAGE ---
            if P.allpos==1
                if S.bc==1
                    pos_buffer(:, :, buffer_idx) = single([p;pgp0.p] / S.rp);
                    % 3. FLUSH TO DISK IF BUFFER FULL
                    if buffer_idx == chunk_size
                        % Determine index range in the file
                        start_idx = qs - chunk_size + 1;
                        end_idx   = qs;                        
                        % Write to disk (Direct access via matfile object)
                        m.POS(:, :, start_idx:end_idx) = pos_buffer;                        
                        % Reset buffer
                        buffer_idx = 1;
                        fprintf('Flushed chunk ending at step %d to disk.\n', qs);
                    else
                        buffer_idx = buffer_idx + 1;
                    end
                else
                    pos_buffer(:, :, buffer_idx) = single(p / S.rp);
                    % 3. FLUSH TO DISK IF BUFFER FULL
                    if buffer_idx == chunk_size
                        % Determine index range in the file
                        start_idx = qs - chunk_size + 1;
                        end_idx   = qs;                        
                        % Write to disk (Direct access via matfile object)
                        m.POS(:, :, start_idx:end_idx) = pos_buffer;                        
                        % Reset buffer
                        buffer_idx = 1;
                        fprintf('Flushed chunk ending at step %d to disk.\n', qs);
                    else
                        buffer_idx = buffer_idx + 1;
                    end
                end
            end
            % ---

            % --- COUNTERS ---            
            qs=qs+1;
            if mod(qs,1e2)==0
                disp({filename,ic,irep,log10(qs)})
            end
            % ---
        end
    end
    clear DATA
end
clearvars -except S V P C