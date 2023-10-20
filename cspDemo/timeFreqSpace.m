function [dataTF,freqs] = timeFreqSpace(data, srate, minfreq, maxfreq, nfreq)
%TIMEFREQSPACE computes time-frequency analysis of the input signals
% Outputs:
%   dataTF - frequency filtered data - analytic signal
%    freqs - frequencies for each frequency channel
% Inputs:
%     data - input data [nb_channels x nb_timestamps]
%    srate - sampling rate /  frequency
%  minfreq - smallest frequency of extracted frequency components
%  maxfreq - highest frequency of extracted frequency components
%    nfreq - number of frequency components extracted
%------------------------------------------
% (c): Peter Rogelj - peter.rogelj@upr.si
%------------------------------------------

if size(data,1)>size(data,2)
    warning('transposing data');
    data=data';
end
if nargin<5
    nfreq=30;
end
if nargin<4
    maxfreq=40;
end
if nargin<3
    minfreq=1;
end
if nargin<2
    warning('Presuming 500Hz sampling rate (or add the 2nd argument).')
    srate=500;
end

nbchan=size(data,1);
pnts=size(data,2);

freqs= logspace(log10(minfreq),log10(maxfreq),nfreq); %generates n points between decades 10^a and 10^b.
Ts=1./freqs;

dataTF=zeros(nbchan, nfreq, pnts);
for nf=1:nfreq %:length(freqs)
    %generate Morlet Wavelet
    b=5*Ts(nf); 
    [cmor,ts] = cmorwavf(-b,b,round(2*b*srate),b*b./5,freqs(nf)); 
    w=sum(abs(real(cmor)));
    cmor=cmor./w;
    %figure(1); plot(ts,[real(cmor);imag(cmor)])
    
    %filter signals
    for ns=1:nbchan
        dataTF(ns,nf,:) = conv(data(ns,:),cmor,'same');
        %figure(nf);plot([ real(filtered); imag(filtered); data(ns,:)]');
        %legend('real','imag','original');  
        %ns
    end
    %nf
end

end

