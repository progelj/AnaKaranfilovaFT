%% for the rource of the testing idea see:
% https://github.com/wmvanvliet/neuroscience_tutorials/blob/master/eeg-bci/3.%20Imagined%20movement.ipynb
% https://www.youtube.com/watch?v=EAQcu6DLAS0


%% load data
clear all;
%load('./data/BCICIV_eval_ds1d.mat');
load('./data/BCICIV_calib_ds1d.mat');
eTimes=mrk.pos;
eTypes=mrk.y;
times=(1:size(cnt,1))/nfo.fs;
nChan=size(cnt,2);

eTimesA=eTimes(eTypes==1);
eTimesB=eTimes(eTypes==-1);

%% frequency distributions ================================================
minfreq=0.5; maxfreq=50; nFreq=30;
[dataTF, freqs]=timeFreqSpace(cnt', nfo.fs, minfreq, maxfreq, nFreq); 
% dataTF consists of analystic sygnals for original data split into frequency components.
% it is organized as [nChannels x nFrequencies x nTimeSamples]
% The real signal is the real component of an analytic signal. 

%% let's see how well our frequency components meet the original data - let's sum all the frequency components
sumTF=squeeze(sum(dataTF,2));
% show data for ch 1;
figure(2);
plot(1:length(sumTF),[cnt(:,1)';real(sumTF(1,:))]);
xticks(eTimes); grid on;
hold on;
plot(eTimesA,zeros(size(eTimesA)),'ro');
plot(eTimesB,zeros(size(eTimesB)),'bo');
hold off;
legend('input', 'sum of freq. comp.', 'events A', 'events B');
title('comparison of the original and the reconstructed signal');

%% illustrate the time-frequency space and the frequency space

electrodesToPlot=[1 2] %1 2 3 4 5 6];
figure(3); 
for nc1=1:length(electrodesToPlot)
    subplot(length(electrodesToPlot),1,nc1); 
    imagesc(squeeze(abs(dataTF(electrodesToPlot(nc1),:,:))),'YData',freqs, 'XData',times); axis xy; set(gca,'YScale','log');
    title(['Freq. distribution of elecrode nr. ' num2str(nc1)]); colorbar;
end

if 0 % set to 1 to plot frequency spectra of all channels
    figure(4); % plot freq. distributions
    for nc1=1:nChan %32
        Fs=sum(abs(dataTF(nc1,:,:)),3);
        subplot(7,9,nc1);
        semilogx(freqs, Fs);
        title(['El. nr. =' num2str(nc1) ]); 
    end
    sgtitle('Freq. distribution for electrodes');
end

%% collect event data (1 second each) -cut out the signals sections at each evente.c
% dataA/dataB, of size (nChan, nTries, nSamples )  - complex (Hilbert transformed)
% TFA/TFB, of size (nChan, nTries, nFreq, nSamples ) - complex
nSamples = 101; % 1 second event length (1s * 100Hz)

%common
Hdata= hilbert(cnt)'; % size (NChan, nSamplesAll) . complex
%TF already computed in dataTF

%event type A
nTriesA = length(eTimesA);
dataA   = zeros(nChan, nTriesA, nSamples);
TFA     = zeros(nChan, nTriesA, nFreq, nSamples);

for iTry=1:nTriesA
    for iChan=1:nChan
        dataA(iChan,iTry,:) = Hdata( iChan, eTimesA(iTry):(eTimesA(iTry)+nSamples-1) );
        TFA(iChan,iTry,:,:) = dataTF( iChan, : , eTimesA(iTry):(eTimesA(iTry)+nSamples-1)  );
    end
end
    
%event type B
nTriesB = length(eTimesB);
dataB   = zeros(nChan, nTriesA, nSamples);
TFB     = zeros(nChan, nTriesB, nFreq, nSamples);

for iTry=1:nTriesB
    for iChan=1:nChan
        dataB(iChan,iTry,:) = Hdata( iChan, eTimesB(iTry):(eTimesB(iTry)+nSamples-1) );
        TFB(iChan,iTry,:,:) = dataTF( iChan, : , eTimesB(iTry):(eTimesB(iTry)+nSamples-1)  );
    end
end

%% check power spectra for electrodes 'C3', 'Cz', 'C4': (ch indexes: 27, 29, 31, see nfo.clab)
nChC3 = find(ismember(nfo.clab,'C3'));
nChCz = find(ismember(nfo.clab,'Cz'));
nChC4 = find(ismember(nfo.clab,'C4'));
C3f = [ squeeze( sum(abs(TFA(nChC3,:,:,:)),[2 4]) )  , squeeze( sum(abs(TFB(nChC3,:,:,:)),[2 4]) ) ];
Czf = [ squeeze( sum(abs(TFA(nChCz,:,:,:)),[2 4]) )  , squeeze( sum(abs(TFB(nChCz,:,:,:)),[2 4]) ) ];
C4f = [ squeeze( sum(abs(TFA(nChC4,:,:,:)),[2 4]) )  , squeeze( sum(abs(TFB(nChC4,:,:,:)),[2 4]) ) ];
figure(10);
subplot(1,3,1);plot(freqs,C3f); legend('L','R'); title('left')
subplot(1,3,2);plot(freqs,Czf); legend('L','R'); title('central')
subplot(1,3,3);plot(freqs,C4f); legend('L','R'); title('right')

%% TEST 1 - following the source solution (logvar features), at f. 8-15 Hz
%input: 3d-array (channels x samples x trials) =>
%    =>  TFA/TFB, of size (nChan, nTries, nFreq, nSamples )  - complex
%output: logvar features (channels x samples x trials) =>
%    => logvarA/logvarB of size: (nChan, nTries)
% 18=8hz, 23=15hz
Fsel=[17:22]
%select only the correct freq. components, from 8 to 15 Hz, according to freqs components 18:23.
sigA= squeeze( sum(real(TFA(:,:,Fsel,:)),3) ); % (nChan, nTries, nSamples ) 
sigB= squeeze( sum(real(TFB(:,:,Fsel,:)),3) );

varA = squeeze( (var(real(sigA),0,3)) );
varB = squeeze( (var(real(sigB),0,3)) );

% May it be better to use average amplitudes of (complex) frequency component?

%% plot logvar features for all channels, both left and right
vars= [ mean(varA,2), mean(varB,2) ];
figure(11); bar(vars,1); legend('left','right');
title('var of each channel');

%%% whitening (sigma) implemented in a function 
%% CSP transform, computes mixing matrix W
sigmasA=zeros(nChan,nChan,nTriesA);
sigmasB=zeros(nChan,nChan,nTriesB);
for iTries=1:nTriesA
    sigmasA(:,:,iTries)=cov( squeeze(sigA(:,iTries,:))' );
end
sigmaA = mean( sigmasA, 3 ); 
for iTries=1:nTriesB
    sigmasB(:,:,iTries)=cov( squeeze(sigB(:,iTries,:))' );
end
sigmaB = mean( sigmasB, 3 ); 


P = whiten(sigmaA + sigmaB);
[U,~,~]=svd( P' * sigmaB * P);
W = P * U;

% uncommnent the next four lines to check the separation :
% L2=W'*sigmaB*W;
% figure(12);subplot(1,2,1);imagesc(L1); colorbar; subplot(1,2,2); imagesc(L2); colorbar
% sgtitle('Eigenvalues after whitening for both event types');

% apply CSP to data
sigCspA=zeros(nChan, nTriesA, nSamples);
sigCspB=zeros(nChan, nTriesB, nSamples);
for iTries=1:nTriesA
    sigCspA(:,iTries,:) = W' * squeeze(sigA(:,iTries,:)); 
end
for iTries=1:nTriesB
    sigCspB(:,iTries,:) = W' * squeeze(sigB(:,iTries,:)); 
end

varCspA = squeeze( log(var(real(sigCspA),0,3)) ); % (nChan, nTriesA);
varCspB = squeeze( log(var(real(sigCspB),0,3)) );

varsCsp= [ mean(varCspA,2), mean(varCspB,2) ];
figure(13); bar(varsCsp,1); legend('left','right');
title('var of each component');

%% Illustrate (plot) x,y separation of cases based on 1st and last component only
figure(14);
c1=1;
c2=59;
plot(varCspA(c1,:),varCspA(c2,:) ,'rx');
hold on;
plot(varCspB(c1,:),varCspB(c2,:) ,'bo');
hold off
legend('left','right')

%% train using Naive Bayes Model

varCspAB=[varCspA,varCspB];
varCspABLabels=[ones(1,100), 2*ones(1,100)];

%selectedComponents=[1 2 3 4 5 6 7 54 55 56 57 58 59];
selectedComponents=[1                            59];

Mdl = fitcnb(  varCspAB(selectedComponents,:)',  varCspABLabels' );

% use the model
label = predict(Mdl,  varCspAB(selectedComponents,:)'  );
confusion = hist3([label,varCspABLabels'],[2 2])
Accuracy = sum( confusion.*eye(2) , 'all') / sum( confusion, 'all')

%% check electrodes with the highest influence:
%
Wcomp = abs(varsCsp(:,2)-varsCsp(:,1))';
Wcomp = Wcomp.^4;
Welectrodes = sum(W.*Wcomp,2);
[WelWeights, WelI]=sort(abs(Welectrodes),1,'descend');
WelNames=nfo.clab(WelI);

