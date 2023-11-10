%% for the rource of the testing idea see:
% https://github.com/wmvanvliet/neuroscience_tutorials/blob/master/eeg-bci/3.%20Imagined%20movement.ipynb
% https://www.youtube.com/watch?v=EAQcu6DLAS0

%% collect event data 

% data from SSVEP - oganized as 
% [nChan=8, nSamples=710, nElectrodeType=2 {dry/wet}, nTry /blocks=10, nTargets=12 {frequencies}]


% TFA/TFB, of size (nChan, nTries, nFreq, nSamples ) - complex
nSamples = 710; % 1 second event length (1s * 100Hz)
nChan = 8;
fs=250;

%TF already computed in dataTF
minfreq=9.25; maxfreq=14.75; nFreq=12;
% it is now possible to define frequencies for which TF is computed:
% minfreq=[9.25 9.75 10.25 10.75 11.25 11.75 12.25 12.75 13.25 13.75 14.25 14.75]; 
if numel(minfreq)>1
    nFreq=numel(minfreq);
end

%event type A
nSubjects=10;
nTries = 10; %length(eTimesA);
TFA     = zeros(nChan, nSubjects*nTries, nFreq, nSamples);
TFB     = zeros(nChan, nSubjects*nTries, nFreq, nSamples);

%event 1 (A)
iCase=1;
for iSubject=1:nSubjects % -all subject %102
    load([ 'dataSSVEP1' filesep 'S' num2str(iSubject,"%03d") '.mat' ] ) % data from SSVEP
    for iTry=1:nTries %10
        eDataA= data(:,:,2,iTry,1);
        [dataTFA, freqs]=timeFreqSpace(eDataA, fs, minfreq, maxfreq, nFreq); 
        eDataB= data(:,:,2,iTry,12);
        [dataTFB, freqs]=timeFreqSpace(eDataB, fs, minfreq, maxfreq, nFreq);
        for iChan=1:nChan
            TFA(iChan,iCase,:,:) = dataTFA( iChan, : , :  );
            TFB(iChan,iCase,:,:) = dataTFB( iChan, : , :  );
        end
        iCase=iCase+1; % case = nsubject * nTry
    end  
end

%% TEST 1 - following the source solution (logvar features), at f. 8-15 Hz
%input: 3d-array (channels x samples x trials) =>
%    =>  TFA/TFB, of size (nChan, nTries, nFreq, nSamples )  - complex
%output: logvar features (channels x samples x trials) =>
%    => logvarA/logvarB of size: (nChan, nTries)
% 18=8hz, 23=15hz
%Fsel=[17:22];
%select only the correct freq. components, from 8 to 15 Hz, according to freqs components 18:23.
tmpTFA= permute(TFA,[1 3 2 4]);
tmpTFA = reshape (tmpTFA, [ size(tmpTFA,1)*size(tmpTFA,2), size(tmpTFA,3), size(tmpTFA,4)] );
tmpTFB= permute(TFB,[1 3 2 4]);
tmpTFB = reshape (tmpTFB, [ size(tmpTFB,1)*size(tmpTFB,2), size(tmpTFB,3), size(tmpTFB,4)] );

sigA= real(tmpTFA(:,:,:)); % (nChan, nTries, nSamples ) 
sigB= real(tmpTFB(:,:,:));

varA = squeeze( (var(real(sigA),0,3)) );
varB = squeeze( (var(real(sigB),0,3)) );

% May it be better to use average amplitudes of (complex) frequency component?

%% plot logvar features for all channels, both left and right
vars= [ mean(varA,2), mean(varB,2) ];
figure(11); bar(vars,1); legend('left','right');
title('var of each channel');

%%% whitening (sigma) implemented in a function 
%% CSP transform, computes mixing matrix W
nFeatures=nChan*nFreq;
sigmasA=zeros(nFeatures,nFeatures,nTries);
sigmasB=zeros(nFeatures,nFeatures,nTries);
for iTries=1:nTries
    sigmasA(:,:,iTries)=cov( squeeze(sigA(:,iTries,:))' );
end
sigmaA = mean( sigmasA, 3 ); 
for iTries=1:nTries
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
sigCspA=zeros(nFeatures, nTries, nSamples);
sigCspB=zeros(nFeatures, nTries, nSamples);
for iTries=1:nTries
    sigCspA(:,iTries,:) = W' * squeeze(sigA(:,iTries,:)); 
end
for iTries=1:nTries
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
c2=8;
plot(varCspA(c1,:),varCspA(c2,:) ,'rx');
hold on;
plot(varCspB(c1,:),varCspB(c2,:) ,'bo');
hold off
legend('left','right')

%% train using Naive Bayes Model

varCspAB=[varCspA,varCspB];
varCspABLabels=[ones(1,10), 2*ones(1,10)];

selectedComponents=[1 : 8];

Mdl = fitcnb(  varCspAB(selectedComponents,:)',  varCspABLabels' );

% use the model
label = predict(Mdl,  varCspAB(selectedComponents,:)'  );
confusion = hist3([label,varCspABLabels'],[2 2])
Accuracy = sum( confusion.*eye(2) , 'all') / sum( confusion, 'all')



