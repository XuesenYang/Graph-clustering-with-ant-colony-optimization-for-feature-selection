function [fit] = sepIndex(train_features, train_targets,vol)
%Reshape the data points using multiple descriminant analysis
%Inputs:
%	train_features	- Input features
%	train_targets	- Input targets
%
%Outputs
%	features	- New features
%	targets	- New targets
%  w	- Weights vector
[D,~]	= size(train_features);
Utargets	= unique(train_targets);
x	= train_features(:,vol);
if (size(train_features,1) < length(Utargets)-1)
   error('Number of classes must be equal to or smaller than input dimension')
end
%Estimate Sw and Sb
m	= mean(x);
Sw_i	= zeros(length(Utargets), D, D);
Sb_i	= zeros(length(Utargets), D, D);
for i = 1:length(Utargets)
   indices	= find(train_targets == Utargets(i)); 
   m_i	= mean(x(indices,:));
   po=zeros(1,1);
   for k=1:length(indices)
       bk=(x(indices(k,1),:)-m_i)*(x(indices(k,1),:)-m_i)';
       po=po+bk;
   end
   Sw_i(i,:,:) = po;
   Sb_i(i,:,:) = length(indices)*(m_i - m)*(m_i - m)';
end
Sw	= squeeze(sum(Sw_i));
Sb	= squeeze(sum(Sb_i));
matrix1=x'*Sb*x;
matrix2=x'*Sw*x;
fit=trace(matrix1)/trace(matrix2);