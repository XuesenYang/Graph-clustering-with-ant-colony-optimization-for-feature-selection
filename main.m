% Editor:Summer
% Institution: Shenzhen University
% E-mail:1348825332@qq.com
% Edit date:2019-2-20
% Title:Integration of graph clustering with ant colony optimization for featur selection
% Sited as: Moradi P., Rostami M.Integration of graph clustering with ant colony optimization for feature selection. Knowledge-Based Systems.2015,84:144-161.DOI: 10.1016/j.knosys.2015.04.007.
%% step1:load datasets and normalize with softmax scaling method
load('sonar.mat');
Theta=0.001;      % Clearance threshold before clustering
gamma=0.2;
maxruns=100;
Ant_numb=50;
epsilon=0.4;
eta=[];           % Intermediate value of pheromone concentration 
X=data(:,1:end-1);
Y=data(:,end);
X_norm=zeros(size(X,1),size(X,2));
r=zeros(size(X,2),size(X,2));
X_var=repmat(var(X),size(X,1),1);
X_mean=repmat(mean(X),size(X,1),1);
group=Y;
class=unique(Y);
for i=1:size(X,1)
    for j=1:size(X,2)
        X_norm(i,j)=1/(1+exp((X_mean(i,j)-X(i,j))/X_var(i,j)));
    end
end
%% step2:caculate Pearson's correlation coefficient
w=corr(X_norm);
w_mean=mean(reshape(w,1,[]));
w_var=var(reshape(w,1,[]));
w_norm=1./(1+exp((w_mean*ones(size(w,1),size(w,1))-w)/w_var));
change_element=w_norm<Theta;
w_norm(change_element)=0; % remove edges which weighs less than ¦È
%% step3:Feature clustering using Louvain-like community detection
A=w_norm;
gamma1 = 1;
k = full(sum(A));
twom = sum(k);
B = full(A - gamma1*k'*k/twom);
[S,Q] = genlouvain(B);  % S identifying the group to which feature has been assigned
S_numb_size=size(unique(S),1);
for i=1:S_numb_size
    Cluster{i}=find(S==i);
end
%% step4:optimization looping 
Tao=gamma*ones(1,size(X,2));                           % Initial pheromone concentration
%-----------------compute Fisher score-----------------%
Fisher_score=Fisher_Score(X,Y);
%------------------------------------------------------%
for i=1:maxruns
    for j=1:Ant_numb
        f=[];
        p=[];
        P=[];
        eta=[];
        F=[];
        Temp=[];
        Select_Cluster=[];
        Traced_Cluster=[];
        Select_Cluster_ID=[];
        Not_Traced_Cluster=[];
        Not_Traced_Number=S_numb_size;
        Not_Traced_Cluster=[1:S_numb_size];
        ID=randperm(Not_Traced_Number);
        Select_Cluster_ID=Not_Traced_Cluster(ID(1));                %The selected class's ID
        Select_Cluster=Cluster{Select_Cluster_ID};                  %current select class
        ID=randperm(length(Select_Cluster));                        %Number of features in selected classes
        f=[f,Select_Cluster(ID(1))];                                %Choose the first feature
        Traced_Cluster=[Select_Cluster_ID];                         %Selected classes
        Not_Traced_Cluster=setdiff([1:S_numb_size],Traced_Cluster); %Unselected classes
        while size(Traced_Cluster,2)<S_numb_size
            if rand>epsilon
                Select_Cluster=Select_Cluster;
                sprintf('remain in old cluster')
            else
                ID=randperm(length(Not_Traced_Cluster));
                Select_Cluster_ID=Not_Traced_Cluster(ID(1));
                Select_Cluster=Cluster{Select_Cluster_ID};
                Traced_Cluster=[Traced_Cluster,Select_Cluster_ID];
                Not_Traced_Cluster=setdiff([1:S_numb_size],Traced_Cluster);
                sprintf('remove to new cluster')
            end
            for fe=1:size(X,2)
                eta(fe)=Fisher_score(fe)-mean(w_norm(fe,f))/size(Traced_Cluster,2);
                F(fe)=eta(fe)*Tao(fe);
            end
            F=F+abs(min(F));
            total_F=sum(F);
            for fe=1:size(X,2)
                p(fe)=F(fe)/total_F;
            end
%--------------------Roulette----------------------%
            P=[];
            Temp=setdiff(Select_Cluster,f);
            for se=1:size(Temp,1)
                P(se)=sum(p(1,Temp(1:se,1)));
            end
            P=P/P(end);
            rand_value=rand;
            ID=1;
                if rand_value>P(ID)
                    ID=ID+1;
                else
                    break
                end      
            f=[f,Temp(ID)];
        end
        val{j}=f;
%-----Evaluate the subset using separability index-------%
        fit(j)=sepIndex(X,Y,val{j});
        det=zeros(1,size(X,2));
        det(1,val{j})=fit(j);
    end
        Tao=0.1*Tao+det; 
end
        [~,n1]=sort(Tao,'descend');   
        Selected_feature=n1(1:S_numb_size*2);
%% step4: caculate classification error rate
e=10;   % e-fold CrossValidation
for i=1:length(class)
    sa=[];
    sa=data((group==class(i)),:);
    [number_of_smile_samples,~] = size(sa); % Column-observation
    smile_subsample_segments1 = round(linspace(1,number_of_smile_samples,e+1)); % indices of subsample segmentation points    
    data_group{i}=sa;
    smile_subsample_segments{i}=smile_subsample_segments1;
end
for i=1:e   
    data_ts=[];data_tr =[];
    for j=1:length(class)
      smile_subsample_segments1=smile_subsample_segments{j};
      sa=data_group{j};
      test= sa(smile_subsample_segments1(i):smile_subsample_segments1(i+1) , :); % current_test_smiles
      data_ts=[test;data_ts] ; % Test data
      train = sa;
      train(smile_subsample_segments1(i):smile_subsample_segments1(i+1),:) = [];
      data_tr =[train;data_tr];% Training data
    end
    mdl = fitcknn(data_tr(:,Selected_feature),data_tr(:,end),'NumNeighbors',4,'Standardize',1);% K nearest neighbor classifier
    Ac1=predict(mdl,data_ts(:,Selected_feature)); 
    Fit(i)=sum(Ac1~=data_ts(:,end))/size(data_ts,1);
end
    fitness=mean(Fit); % classification error rate

