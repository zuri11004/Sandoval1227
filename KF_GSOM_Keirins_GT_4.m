%zurisaddai Sandoval Lara
%Tesis: Self-organizing clustering  by Growing-SOM for EEG Biometrics
%Asesora: Dra. Pilar Gómez Gil
%modificado: 26 de abril 2020.
%Programa que lee las bases de datos de los conjuntos testing y training
%del conjunto con filtro chebychev de orden 2 y crea una matriz refinada
%por la red som
%and validation y impostor
%por medio de funciones realiza la verificacion . refinando las
%caracteristicas  usando GSOM normalizacion de las carecterisiticas
%extraidas por DWT por Db4
%clasifcación por medio de mlp.

%-----------------menu----------------------------------
clear all
clc
while true
disp('opcion [1] : cargar base de datos')
disp('opcion [2] : Entrenamiento de GSOM  ')
disp('opcion [3] : Matriz refinada GSOM  ')
disp('opcion [4] : elaboracion de conjuntos de datos para plantillas por sujeto para mlp')
disp('opcion [5] : Entrenar MLP  ')
disp('opcion [6] : Se crea el conjunto de validacion (validation+impost) with GTrue')
disp('opcion [7] : verificador y metricas  ')
disp('opcion [8] : Graficador de metricas')
disp('salir [otro numero]')
opcion=input('ingrese opcion:');
 
switch(opcion)
case 1
    disp('cargar base de datos de entrenamiento normalizada y testing normalizada y validation normalizada')
    %____________________cambiar nombre por la tarea deseada
    Base_Datos1=load('Matriz_norm_keirns_training_c5_baseline'); % se carga la matriz original con sus etiquetas
    Base_Datos2=load('Matriz_norm_keirns_testing_c5_baseline'); % se carga la matriz original con sus etiquetas
    Base_Datos4=load('Matriz_norm_keirns_impost_c5_baseline.mat');% se carga la matriz original con sus etiquetas
    %_______________________hasta aqui
    Matriz_basedatos_training=Base_Datos1.data;
    Matriz_basedatos_testing=Base_Datos2.data;
    Matriz_basedatos_impost=Base_Datos4.data;
    %tamaño de las bases de datos
    [~, ca_N] =size(Matriz_basedatos_training); 
    [~,ca_N1] =size(Matriz_basedatos_testing);  
    [~,ca_N3] =size(Matriz_basedatos_impost);

    %_______________se tienen las tres matrices normalizadas
    Mtarget_train= Matriz_basedatos_training(:,ca_N); % Vector de etiqueta
    Mtrain_training= Matriz_basedatos_training(:,1:(ca_N-1)); %Matriz de entrenamiento con caracteristicas
    Mtarget_test= Matriz_basedatos_testing(:,ca_N1); % Vector de etiqueta
    Mtrain_testing= Matriz_basedatos_testing(:,1:(ca_N1 -1)); %Matriz de entrenamiento con caracteristicas
    Mtarget_impost= Matriz_basedatos_impost(:,ca_N3); % Vector de etiqueta
    Mtrain_impost= Matriz_basedatos_impost(:,1:(ca_N3 -1)); %Matriz de entrenamiento con caracteristicas
    
   %_______________se tienen cargadas las bases de datos
    case 2
        %Entrenamiento de la red Gsom
        disp('Entrenamiento de la GSOM solo una vez')
        load('pre_param_c5_baseline');
        [~,col]=size(Mtrain_training);
        data1=Mtrain_training;
        class=Mtarget_train;

        %Configura la red neuronal de G_SOM
         netset = setting('gg');     
         pre_param1=pre_param;
         net=gg(data1,netset,pre_param1); %Entrenamiento de la red GSOM
         netGsom=net;
         
         %Vizualisacion de hip map
         %figure(1);vismap(net,class) %
         %figure(2);vishit(net)
        
        %_______________guarda red som
         fprintf('Que tarea escogiste? opciones \n letter-composing \n baseline\n');
         fprintf(' multiplication \n rotation \n counting\n');
         tarea = input('escribe entre comillas simples la tarea ');
         nombre1=cat(2,'netGsom_',tarea);
         save(nombre1,'netGsom');
 
       case 3
           %se refinan los datos de los tres conjuntos
           %se carga la red entrenada
           load('netGsom_baseline.mat');
           
          % ____________________________
        MRefinada= RefGSOM(Mtrain_training,netGsom.W,Mtarget_train);
        MRefinada_training = MRefinada;
        MRefinada= RefGSOM(Mtrain_testing,netGsom.W,Mtarget_test);
          %Refina la matriz de prueba
        MRefinada_testing = MRefinada;
        MRefinada= RefGSOM(Mtrain_impost,netGsom.W,Mtarget_impost);
        MRefinada_impost = MRefinada;
   
 % se guarda  la base de dato refinada de training y testing e impost
%_________________________________________
         fprintf('Que tarea escogiste? opciones \n letter-composing \n baseline\n');
         fprintf(' multiplication \n rotation \n counting\n');
         tarea = input('escribe entre comillas simples la tarea  ');
         nombre1=cat(2,'Matriz_refinadaG_KEIRN_training_',tarea);
         nombre2=cat(2,'Matriz_refinadaG_KEIRN_testing_',tarea);
         nombre4=cat(2,'Matriz_refinadaG_KEIRN_impost_',tarea);
         save(nombre1,'MRefinada_training');
         save(nombre2,'MRefinada_testing');
         save(nombre4,'MRefinada_impost');
   
           %Mtrain_training=MRefinada_training
           %Mtrain_impost=MRefinada_testing
           %Mtrain_testing=MRefinada_impost
   
            case 4
              load('Matriz_refinadaG_KEIRN_training_baseline');
              [~,rc]=size(MRefinada_training);
              Mtarget_train=MRefinada_training(:,rc);
              Mtrain_training=MRefinada_training(:,1:(rc-1));
 
             %______________contar cuantas personas hay por sujeto
              [s1,s3,s4,s5,s6] = contar(Mtarget_train);
              
              vec_valores=horzcat(s1,s3,s4,s5,s6);
              max_con=min(vec_valores); %se obtien el valor maximo por cada conjunto para crear conjuntos balanceados para el entrenamiento
              %_________________________________
              data_training=[Mtrain_training Mtarget_train];
              %de los conjuntos refinado de training crear 5 conjuntos para los 5 clasificadores que seran las plantillas de cada sujeto
              fprintf('Que tarea escogiste? opciones \n letter-composing \n baseline\n');
              fprintf(' multiplication \n rotation \n counting\n');
              tarea = input('escribe entre comillas simples la tarea  '); 
              %______conjunto sujeto 1
              
              [s1_t,s3_t,s4_t,s5_t,s6_t]=Generar_conjuntos(data_training,max_con,s1,s3,s4,s5,s6);

              nombre=cat(2,'MCS1_KEIRN_training_',tarea);
              fprintf('Nombre del nuevo archivo: %s \n', nombre);
              save(nombre,'s1_t');
              
               %______conjunto sujeto 3
              
              
              
              nombre=cat(2,'MCS3_KEIRN_training_',tarea);
              fprintf('Nombre del nuevo archivo: %s \n', nombre);
              save(nombre,'s3_t');
              
               %______conjunto sujeto 4
             
             
              
              nombre=cat(2,'MCS4_KEIRN_training_',tarea);
              fprintf('Nombre del nuevo archivo: %s \n', nombre);
              save(nombre,'s4_t');
               %______conjunto sujeto 5
               
              
              nombre=cat(2,'MCS5_KEIRN_training_',tarea);
              fprintf('Nombre del nuevo archivo: %s \n', nombre);
              save(nombre,'s5_t');
               %______conjunto sujeto 6 son todas las clases balanceadas
               
              
              
              nombre=cat(2,'MCS6_KEIRN_training_',tarea);
              fprintf('Nombre del nuevo archivo: %s \n', nombre);
              save(nombre,'s6_t');
%              
                case 5
                    %_____________________Entrena una red con las
                    %plantillas de datos generadas 
                     disp('Entrenamiento de MLP') 
                     fprintf('Que tarea escogiste? opciones \n letter-composing \n baseline\n');
                     fprintf(' multiplication \n rotation \n counting\n');
                     tarea = input('escribe entre comillas simples la tarea ');
                  %______________cambiar nombre por tarea deseada
                  MRTS1=load('MCS1_KEIRN_training_baseline'); %se carga conjunto para cada sujeto 
                  MRTS3=load('MCS3_KEIRN_training_baseline'); %se carga conjunto para cada sujeto
                  MRTS4=load('MCS4_KEIRN_training_baseline'); %se carga conjunto para cada sujeto
                  MRTS5=load('MCS5_KEIRN_training_baseline'); %se carga conjunto para cada sujeto
                  MRTS6=load('MCS6_KEIRN_training_baseline'); %se carga conjunto para cada sujeto
                  %________________________________________________________________________________
                  %MLP Manual % se realizara una plantilla para cada sujeto
                  %por lo cualse tendra 5 mlp por sujeto cambiando en la
                  %ultima capa una funcionde transferencia tipo softmax
                   neur=input('número de neuronas \n');
                   Epocas=input('número de epocas a entrenar \n');
                   netmlpS1=mlp(MRTS1.s1_t,neur,Epocas);
                   
                   %Se gurdan las plantillas de los sujetos
                   %que corresponden a las redes mlp para cada sujeto
                   nombre1=cat(2,'netmlpS1_',tarea);
                   save(nombre1,'netmlpS1');
                   %______________sujeto 3
                   netmlpS3=mlp(MRTS3.s3_t,neur,Epocas);
                   nombre1=cat(2,'netmlpS3_',tarea);
                   save(nombre1,'netmlpS3');
                   %___________sujeto 4
                   netmlpS4=mlp(MRTS4.s4_t,neur,Epocas);
                   nombre1=cat(2,'netmlpS4_',tarea);
                   save(nombre1,'netmlpS4');
                   %___________sujeto 5
                   netmlpS5=mlp(MRTS5.s5_t,neur,Epocas);
                   nombre1=cat(2,'netmlpS5_',tarea);
                   save(nombre1,'netmlpS5');
                    %___________sujeto 6
                   netmlpS6=mlp(MRTS6.s6_t,neur,Epocas);
                   nombre1=cat(2,'netmlpS6_',tarea);
                   save(nombre1,'netmlpS6');                  
                   
                     case 6
                         %__________se crea la testing con impostores
                         load('Matriz_refinadaG_KEIRN_testing_baseline');
                          [~,rc]=size(MRefinada_testing);
                          Mtarget_test=MRefinada_testing(:,rc);
                          M_testing=MRefinada_testing;  
                         load('Matriz_refinadaG_KEIRN_impost_baseline');
                          [~,rc]=size(MRefinada_impost);
                          Mtarget_impost=MRefinada_impost(:,rc);
                          M_impost=MRefinada_impost;
                          
                          
                          [p,~] = size(M_testing);
                          GT=ones(p,1);
                          M_test=[M_testing GT]; 
                          [p,q] = size(M_impost);
                          vec_sustituto = ones(p,1);
                           for i=1:p
                               r1 = randi(6);
                                if(r1==2)
                                  vec_sustituto(i,1)=vec_sustituto(i,1);
                                else
                                  vec_sustituto(i,1)=r1;
                                end
                           end
                           M_impost(:,q)=vec_sustituto(:,1);
                           GF=zeros(p,1);
                           M_imp=[M_impost GF];
                           
                           M_prueba=[M_test;M_imp];
                           
                            
                           fprintf('Que tarea escogiste? opciones \n letter-composing \n baseline\n');
                           fprintf(' multiplication \n rotation \n counting\n');
                           tarea = input('escribe entre comillas simples la tarea ');
                           nombre=cat(2,'Matriz_refinadaG_KEIRN_testimp_',tarea);
                           fprintf('Nombre del nuevo archivo: %s \n', nombre);
                           save(nombre,'M_prueba');
                         
                      case 7
                          %se cargan la matriz de refinamiento de
                          %validation para pobrar en la plantilla que dice
                          %ser
                          %__________-cambiar nombre por la tarea deseada
                          load('Matriz_refinadaG_KEIRN_testimp_baseline');
                          
                          %______________se cargan las plantillas de los
                          %sujetos permitidos
                          load('netmlpS1_baseline');
                          load('netmlpS3_baseline');
                          load('netmlpS4_baseline');
                          load('netmlpS5_baseline');
                          load('netmlpS6_baseline');
                          %________________________________
                          %_________hasta aqui
                          MRefinada_validation1=M_prueba;
                          
                          disp('Verificador de MLP y metricas')
                          [Nsuj_aux Nu_car] = size(MRefinada_validation1); %tamaño del conjunto validacion para las iteraciones
                           Matriz_excel=zeros(Nsuj_aux,8);
                           umbral=0: 0.01: 1;
                           %umbral = [0.0 0.05 0.1 0.15 0.20 0.25  0.30  0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75  0.80  0.85 0.90 0.95 1.0];%modificara
                           % umbrales definidos de 0.1 de incremento
                           % umbral=[1.5]; 
                            [f_umbra,c_umbral] = size(umbral); % tamaño del umbral para 
                            FRR=zeros([1 c_umbral]);
                            FAR=zeros([1 c_umbral]);%______________________metricas
                            
                            
                            
                        for i_umbral=1:c_umbral
                            um_ele=umbral(1,i_umbral); %se decide con que umbral trabajar
                            TP=0;
                            FP=0;
                            TN=0;
                            FN=0;
                            for i=1:Nsuj_aux
                               
               
                                   suj_aux= MRefinada_validation1(i,1:Nu_car-2); %vector de entrenamiento con caracteristicas refinas del conjunto validacion
                                   et_aux= MRefinada_validation1(i,Nu_car-1); % etiqueta auxiliar
                                   gt_identi= MRefinada_validation1(i,Nu_car); %grand true auxiliar 
                                   
                                   
                                   
                                   %ya que tenemos la etiqueta reclamamos
                                   %la identidad en la plantilla deseada y
                                   %permitida y el grand true elegimos al
                                   %sujeto
                                   z = transpose(suj_aux); %Transpuesta del vector a evaluar en la red  

                                    
                          %--------------------------------- se evalua en la red del sujeto 1

                                            if(et_aux== 1)
                                               netmlp=netmlpS1;
                                               
                                            end
                                            if(et_aux== 3)
                                               netmlp=netmlpS3;
                                            end
                                            if(et_aux== 4)
                                               netmlp=netmlpS4;
                                            end
                                            if(et_aux== 5)
                                               netmlp=netmlpS5;
                                            end  
                                            if(et_aux== 6)
                                               netmlp=netmlpS6;
                                            end                                          
                                           % Matriz_excel(i,1)=et_aux; %posicion 1 indica el sujeto seleccionado
                                          %  Matriz_excel(i,2)=gt_identi;
                                             y = netmlp(z); %se evalua el vector de validacion en la red neuronal                             
                                             Vector_salida = y; %obtiene los valores de la capa de salida
                                              %Matriz_excel(i,3)=Vector_salida(1,1); % posicion 2 indica el Y en la posicion 1
                                             if(Vector_salida(1,1)>=um_ele)
                                                 y_selec=1;
                                             else
                                                  y_selec=0;
                                             end
                                               % Matriz_excel(i,4)=y_selec; %posicion 3 se guarda el Y seleccionado si es mayor que el unbral
                                                 
                                                
                                              if(y_selec==1 && gt_identi==1) %true positive
                                                  TP=TP+1;
                                                 % Matriz_excel(i,5)=1; %posicion 4 se guarda el TP
                                              end

                                       
                                              if(y_selec==0 && gt_identi==0) %true negative
                                                 TN=TN+1;
                                               %  Matriz_excel(i,6)=1; %posicion 5 se guarda el TN
                                              end

                                              if(y_selec==1 && gt_identi==0) %False Positive
                                                  FP=FP+1;
                                                 % Matriz_excel(i,7)=1; %posicion 6 se guarda el FP
                                              end
                                              
                                              if(y_selec==0 && gt_identi==1) %False negative
                                                  FN=FN+1;
                                                 % Matriz_excel(i,8)=1; %posicion 4 se guarda el FN
                                              end
                                       
          
                                   
                          
                            end%-------------evalua los sujetos de validacion de acuerdo al umbral i
                            %metrica
                            %xlswrite('excel_baseline01.xlsx', Matriz_excel, 'Hoja1','A2');
                            metrica_far=FP/(FP + TN);
                            FAR(1,i_umbral)=metrica_far; %Taza de falsa aceptacion
                            metrica_frr=FN/(TP + FN);
                            FRR(1,i_umbral)=metrica_frr; %Taza de falso rechazo
                            
                        end
                          Umbral_transp=umbral';
                            
                            case 8
                                  disp('Graficador')
                                  X1=Umbral_transp;
                                  Y1=FAR;
                                  Y2=FRR;
                                  
                                  figure
                                  plot (X1,Y1, X1,Y2);
                                  title('EER')
                                  xlabel('Threshold(Umbral)')
                                  ylabel('FAR & FRR') 
                                  legend({'Y1 = FAR','Y2 = FRR'},'Location','southwest')
                               %--------------------------------obtener la
                                   otherwise
                                   break
 end
 
end











%--------------------Funciones--------------------
  function [s1,s3,s4,s5,s6] = contar(Matriz)
  %Funcion que te cuenta cuantos sujetos hay para cada sujeto en el
  %conjunto training
         [Nu_suj,~]=size(Matriz);
         s1=0;
         s3=0;
         s4=0;
         s5=0;
         s6=0;
         for i=1:Nu_suj
                    if (Matriz(i,1)==1) 
                     s1 = s1+1;
                    end
                    if (Matriz(i,1)==3) 
                     s3 = s3+1;
                    end
                    if (Matriz(i,1)==4) 
                     s4 = s4+1;
                    end
                    if (Matriz(i,1)==5) 
                     s5 = s5+1;
                    end
                    if (Matriz(i,1)==6) 
                     s6 = s6+1;
                    end

         end  
  end
  

function [s1_t,s3_t,s4_t,s5_t,s6_t]=Generar_conjuntos(data_training,max_con,s1,s3,s4,s5,s6)
              
              
              %________etiquetas solo se calcula una sola vez
              aux1=ones(1,max_con);
              aux2=1+ aux1;
              VT_aux=horzcat(aux1,aux2); %donde 1 es la clase del sujeto 1 y 2 las clases de los sujetos no aceptados
              [~,sc] = size(data_training);
              %______________________________
              %______________sujeto1
              s1_train=data_training(1:(0+s1),1:(sc-1));
              sr_train=data_training(:,1:(sc-1));
              sr_train(1:(0+s1),:)=[]; %se quita  todo el conjunto del sujeto 1
              rndid=randperm(size(sr_train,1)); %revuelve al azar el conjunto
              sr_train=sr_train(rndid,:);
              
              aux1_t=s1_train(1:max_con,:);
              aux2_t=sr_train(1:max_con,:);
              VT_aux=VT_aux';          
              s1_t=[aux1_t;aux2_t];
              s1_t=[s1_t VT_aux];
              %________________sujeto3
              rang_fil=s1+1;
              rang_fil2=s1+s3;
              s1_train=data_training(rang_fil:rang_fil2,1:(sc-1));
              sr_train=data_training(:,1:(sc-1));
              sr_train(rang_fil:rang_fil2,:)=[]; %se quita  todo el conjunto del sujeto 1
              rndid=randperm(size(sr_train,1)); %revuelve al azar el conjunto
              sr_train=sr_train(rndid,:);
              
              aux1_t=s1_train(1:max_con,:);
              aux2_t=sr_train(1:max_con,:);         
              s3_t=[aux1_t;aux2_t];
              s3_t=[s3_t VT_aux];
              %________________sujeto4
              rang_fil=rang_fil+s3;
              rang_fil2=rang_fil2 + s4;
              s1_train=data_training(rang_fil:rang_fil2,1:(sc-1));
              sr_train=data_training(:,1:(sc-1));
              sr_train(rang_fil:rang_fil2,:)=[]; %se quita  todo el conjunto del sujeto 1
              rndid=randperm(size(sr_train,1)); %revuelve al azar el conjunto
              sr_train=sr_train(rndid,:);
              
              aux1_t=s1_train(1:max_con,:);
              aux2_t=sr_train(1:max_con,:);         
              s4_t=[aux1_t;aux2_t];
              s4_t=[s4_t VT_aux];
              %________________sujeto5
              rang_fil=rang_fil+s4;
              rang_fil2=rang_fil2 + s5;
              s1_train=data_training(rang_fil:rang_fil2,1:(sc-1));
              sr_train=data_training(:,1:(sc-1));
              sr_train(rang_fil:rang_fil2,:)=[]; %se quita  todo el conjunto del sujeto 1
              rndid=randperm(size(sr_train,1)); %revuelve al azar el conjunto
              sr_train=sr_train(rndid,:);
              
              aux1_t=s1_train(1:max_con,:);
              aux2_t=sr_train(1:max_con,:);       
              s5_t=[aux1_t;aux2_t];
              s5_t=[s5_t VT_aux];
              %________________sujeto6
              rang_fil=rang_fil+s5;
              rang_fil2=rang_fil2 + s6;
              s1_train=data_training(rang_fil:rang_fil2,1:(sc-1));
              sr_train=data_training(:,1:(sc-1));
              sr_train(rang_fil:rang_fil2,:)=[]; %se quita  todo el conjunto del sujeto 1
              rndid=randperm(size(sr_train,1)); %revuelve al azar el conjunto
              sr_train=sr_train(rndid,:);
              
              aux1_t=s1_train(1:max_con,:);
              aux2_t=sr_train(1:max_con,:);      
              s6_t=[aux1_t;aux2_t];
              s6_t=[s6_t VT_aux];
              %________________Fin del proceso
              
              
              
end
 
function net=mlp(data,neur,Epocas)
                   [~, Nu_c] = size(data);
                   Mtargets= data(:,Nu_c);
                   Mtargets = dummyvar(Mtargets);
                   Mtrainref= data(:,1:Nu_c-1); %Matriz de entrenamiento con caracteristicas refinadas
                   xs = Mtrainref';
                   ts = Mtargets';
                   net = feedforwardnet(neur,'trainscg');
                   net.layers{2}.transferFcn='softmax';
                   net.outputs{end}.processFcns={ }; 
                   net.trainParam.epochs = Epocas;
                   net.divideFcn = ''; % el conjunto de entrenamiento es el 100 porciento
                   net.trainParam.lr = 0.1;
                   net.trainParam.goal = 0.0;
                   net = configure(net, xs, ts); %se configura la red para el entrenamiento con matriz  x de entrada y t1 matriz etiquetas del conjunto
                   net = init(net); %incializa los pesos
                  fprintf('Se entrena la red .\n');
                  [net,~] = train(net,xs,ts); %se entrena la red neuronal con todo el conjunto de la matriz trainning normalizada y refinada
end

function net=SOM(data)
        
        Ptrain = transpose(data); % Matriz  a entrenar
        N_neuronas=input('número n de neuronas para crear neuronas iniciales nxm neuronas \n');
        M_neuronas=input('número m de neuronas para crear neuronas iniciales nxm neuronas \n');
        Epocas=input('número de epocas a entrenar \n');
        net = selforgmap([N_neuronas M_neuronas]); % . se crea una red . de nxm neuronas.
        fprintf('Se crea una red de  %d X %d neuronas .\n',N_neuronas,M_neuronas);
        net = configure(net,Ptrain);
        % Puede configurar la red para ingresar los datos y trazar 
        %Al simular una red, las distancias negativas entre el vector de peso de cada neurona
        %y el vector de entrada se calcula (negdist) para obtener las entradas ponderadas. El ponderado
        %las entradas son también las entradas netas (netsum). Las entradas netas compiten (competencia) de modo que solo el
        %La neurona con la entrada neta más positiva generará un 1.
        net.trainParam.epochs = Epocas;
        fprintf('Se entrena la red som con   %d epocas .\n',Epocas);
        net = train(net,Ptrain); % entrenamie
end


%______________________ Funciones para la GSOM 
function MRefinada= RefGSOM(data1,W,class)
%Funcion que Refina con GSOM
     [p,q]=size(data1);
     MRefinada=zeros(p,q);

     for i=1:p
     data_tmp=data1(i,:);
     [~,winlist]=min(dist(W',(data_tmp)'));
     w_asoc=W(:,winlist);
     Vec_BMU=w_asoc';
     MRefinada(i,:) = Vec_BMU;
     end
     MRefinada = [MRefinada class];
   %_________Explicacion  
     
 %______comienza el experimento para refinar
%_______lee el conjunto de datos data1
%______se carga la red que seria la GSOM=struc(net)
%[p,q]=size(data1)
%MRefinada=zeros(p,q)
%_____se carga vector clase class
%Evalua el vector y te saca la neurona ganadora y toma su vector de pesos asociado
%for i=1:p
%[Vec_pesos,winlist]=winfun(data_tmp(i,:),net.W);
%se asigna a la posicion i del vector refinado
%Vec_aux=Vec_pesos'
%MRefinada(i,:) = Vec_aux;
%end
%MRefinada = [MRefinada class];
%Termina el proceso de la matriz refinada
end



%_________________funciones de G-SON
function netset=setting(method)
%El metodo es GG que significa Growing Grid = G-SOM
% La configuración  inicializa una estructura predeterminada para los parámetros
netset.method = method;    % metodo G-som
epoc=input('N. de epocas');
netset.epch   = epoc;       % Numero de epocas
netset.amax   = 0.2;       % tasa de aprendizaje maximo
netset.amin   = 0.05;      % tasa de aprendizaje minimo
netset.neighb = 'gauss';  % Funcion de vecindad = Gaussiana por definicio de Fritze
netset.vis    = 'n';       % solo se visualisa si es de 2 dimensiones
netset.inw    = 'rn';      % inicializar los pesos aleatoriamente
netset.pmax   = 7.5;         % max. radio vecino
netset.pmin   = 0.7;         % min. radio vecino
netset.sf     = 0.8;       % spread-out factor
end
function net=gg(data,netset,pre_param1)
% Función de entrenamiento
[n1,m1]=size(data);
%Tamaño de la matriz de datos
data_tmp=data; 
%Datos normalizados entre cero y 1
pre_param=pre_param1;
%Vector de valores minímos y maxímos para la normalización
net=gg_init(data,netset); 
%inicializacion de pesos aleatorios para las 4 neuronas iniciales 
%y sus posiciones de las neuronas.

ds=linkdist(net.grd); 
% saca la distacia con cada neurona que esta en la rejilla                                                              %[0,1,1,2] para las 4 establecidad 1-1=0dist 1-2=1 dis 1-3=1 dist y la %1-4=2 dista, y asi para el resto dependiendo del tamaño de neuronasr
epvec=1:netset.epch; 
% vector de epocas totales                                                                                              (1:100) numero total en vector
netset.pvec=(netset.pmax-netset.pmin)*((netset.epch-epvec)/(netset.epch-1))+netset.pmin;
% actualizacion de la disminuncion de radio vecino por cada iteracion
netset.lrvec=(netset.amax-netset.amin)*((netset.epch-epvec)/(netset.epch-1))+netset.amin;
%actualizacion de taza de aprendizaje minimizando por cada iteracion                                                       o epoca, se establece un vector y un valor por cada epoca
netset.GT=-m1*log(netset.sf); 
%este umbral de crecimiento de acuerdo  al max error acumulado que puede
%tener una neurona                                                                                                       se obtiene por el factor de dispecion dado y el numero %de caracterisitcas de los datos se establece  GT
%Empieza el aprendizaje
fprintf('Growing SOM  \nTraining ...')
for ep=1:netset.epch 
    %Recorre el numero de epocas(iteraciones)
    net.nnum(ep)=size(net.W,2); 
    %numero de neuronas de la epoca actual                                                                               %por cada iteracion  %rndid=randperm(size(data,1)); %crea un vector del tamaño de elementos filas de nuestra base de datos  con posiciones enteras aleatorias es decir [1, 158, 4, ...] solo posiciones sin repetirse %data=data(rndid,:); %lo anterior se crea para cambiar el orden de los datos de manera aleatoria y sea menos un sesgo de eleccion.
    net.Err=zeros(1,size(net.grd,2)); 
    %inicializa el error por cada posicion de neurona en 0, es decir,
    %si comienza con 4 neuronas tendra un vector de error de tamañano
    %[0 0 0 0] donde cada elemento al iniciar tiene error 0.
    h=nbrfun(netset,ds,ep); 
    %Calcula la fase de crecimiento por medio de la funcion gausiana de la epoca 1
    for sm=1:n1 
        %For de tamaño de filas de los datos. 
        %En esta parte se escoje la neurona ganadora para cada entrada 
        [~,win_id]=winfun(data(sm,:),net.W);
        % devuelve la posicion de la BMU (Neurona ganadora)                                                             %       h=nbrfun(netset,ds,ep);
%_____proceso donde se realiza w_i(t+1)=w_i(t) + n(t)*h(t)*(x_j - w_i(t))
        DW=repmat(h(win_id,:),size(net.W,1),1).*(repmat(data(sm,:)',1,size(net.W,2))-net.W); 
        %se hace el calculo de la funcionde vecindad de la neurona ganadora
        %con la fase de aprendizaje de esa iteracion
        net.W=net.W+DW; 
        %actualizacion de pesos, es decir, adaptan los nuevos pesos
        %de la neurona ganadora y sus vecinos
 %_____________________________fin de este proceso       
        %calcula el error de cada neurona                                                                                 %recordemos que el error acumulativo de la neurona ganadora  %calcula y se actualiza de acuerdo con la distancia entre la señal %de entrada y el vector de pesos ganador
        [win_Err,win_id]=winfun(data(sm,:),net.W); 
        % se obtine el error de de la ganadora, definido como el
        %minimo valor de las distancias entre neuronas y la posicion de
        %la neurona ganadora E_winner(t+1)= E_winner(t) + ||x_j - w_winner||
        net.Err(win_id)=net.Err(win_id)+win_Err; 
        % error acumulativo de neurona ganadora
        
        % Crecimiento de rejilla
        if net.Err(win_id)>=netset.GT
            %se usa maxímo error acumulativo que puede tener una neurona
            %y se define como GT.                                                                                       %se pregunta si mi error acumulativo de mi neurona ganadora es %mayor que el error maximo que puede tener cada neurona
            nbr_id=ds(win_id,:)==1;
            %se obtienen los vecinos de la neurona ganadona de acuerdo a
            %la distancia entre ellos. 
            [~,far_id]=max(nbr_id.*dist(net.W(:,win_id)',net.W));                                                         %se calcula la distancia de pesos de la neurona ganadora con %sus vecinas obteniendo la poscion de la neurona mas lejanada  %vecina de la neurona ganadora
            %indica la posicion de la unidad mas lejana
            shift_vec=net.grd(:,win_id)-net.grd(:,far_id); 
            %esta funcion te indica las coordenadas a donde se insertara la
            %nueva neurona que es la direccion de crecimiento , la posicion
            %1 es a lo largo del eje x y la posicion 2 es a lo largo del
            %eje y.
            grow_dir=find(abs(shift_vec)>0);
            %indica la posicion del eje de insercion           
            new_cut_ids=find(net.grd(grow_dir,:)==net.grd(grow_dir,win_id));
            %posicion de las unidades frontera en movimiento
            new_grds = net.grd(:,new_cut_ids); 
            % se selecciona el conjunto de las posicion de las neuronas
            %donde seran los limites de la neurona insertada.
            adj_grds=net.grd(:,new_cut_ids)-repmat(shift_vec,1,length(new_cut_ids));
            %posicion de las neuronas sobrantes en la rejilla
            [~,adj_pos]=ismember(adj_grds',net.grd','rows');
            %procedimiento de insercion de la nueva neurona donde usa 
            %ismember que es una funcion de elementos de matriz que 
            %son miembros de la matriz establecida
            % se obtiene la nueva posicion de la neurona 
            
            
            
            
            W_new=(net.W(:,new_cut_ids)+net.W(:,adj_pos))/2;
            % Se asigna su vector de pesos a las nuevas neuronas agredas
            net.W=[net.W,W_new]; % pesos agregados nuevos                                                                   %              net.Err(new_cut_ids) = net.Err(new_cut_ids)/2  %             net.Err(adj_pos) = net.Err(adj_pos)/2;
            %proceso de distribucion de error
            net.Err(new_cut_ids) = 0;
            net.Err(adj_pos) = 0;
            net.Err=[net.Err,zeros(1,length(new_cut_ids))];
            %se inicializa en 0 ahora los errores asociadosa las nuevas
            %neuronas. 
            %___insercion y modificacion para agregar las nuevas posiciones de las neuronas
            if sum(shift_vec)>0
                move_ids=find(net.grd(grow_dir,:)>=net.grd(grow_dir,win_id));
            else
                move_ids=find(net.grd(grow_dir,:)<=net.grd(grow_dir,win_id));
            end
            net.grd(:,move_ids)=net.grd(:,move_ids)+repmat(shift_vec,1,length(move_ids));
            net.grd=[net.grd,new_grds];
            %________________aqui termina de actualizarse la posicion de
            %las neuronas nuevas en las rejillas
            ds=linkdist(net.grd); 
            % se obtiene ahora la nueva distancia entre cada una de ellas
            h=nbrfun(netset,ds,ep);  
            %se calcula la fase de crecimiento por medio de 
            %la funcion gausiana de la epoca i para las nuevas neuronas
        end
    end
    %fprintf('[termina el entrenamiento de datos]\n')
    
    
    
    
    % Finalmente se grafica las neuronas en el espacio de datos
    %siempre y cuando tenga la opcion de visualizar  
    if strcmp(netset.vis,'y') %se pregunta en caso de ser si
        vistrn(net,data) % esta funcion visualiza
    end
end %_______________________________________________
    
fprintf('[termina el entrenamiento de datos]\n')

net.grd=net.grd-repmat(min(net.grd')',1,size(net.grd,2));%vector de posiciones de todas las neuronas en las rejillas

for i=1:n1
    [~,winlist(i)]=winfun(data_tmp(i,:),net.W); %lista de neuronas ganadoras asociadas a cada vector de entrada de todas las neuronas
end

hitcount=zeros(1,size(net.W,2)); %vector de 98 posiciones correspondiente a todas las neuronas y sus pesos asociados 
for j=1:size(net.W,2)
    hitcount(j)=sum(winlist==j);
end % para crear el mapa de hit indica el numero de nuronas correspondiente a cada possicion del vector 

net.winlist=winlist; %se agraga a la estructura net la lista
net.hitcount=hitcount; % es el histograma de aciertos de la red som
net.pre_param=pre_param; %Guarda los maxmos y minimos para la normalizacion en net
end

function vistrn(net,data)
% grafica las neuronas en el pacio de datos si es de dos dimensiones

[xcord,ycord]=find(triu(linkdist(net.grd))==1);
figure(5)
plot(data(:,1),data(:,2),'.','markersize',12)
hold on
plot(net.W(1,:),net.W(2,:),'.r','markersize',18,'markerfacecolor','r')
for ii=1:length(xcord)
    line([net.W(1,xcord(ii)),net.W(1,ycord(ii))],[net.W(2,xcord(ii)),net.W(2,ycord(ii))],'Color','r');
end
axis equal
hold off
drawnow
end %______________________grafica la red en la misma figura para reaizarla dinamica solo en caso de tener datos en R2



function [Minval,winlist]=winfun(data,W)
%Encuentra la lista de neuronas donde indica la  neurona ganadora y valores de error correspondientes para datos
% cuando se presenta un señal como la som para cada señal de entrada se le
% resta los pesos de cada neurona, de esa lista la distancia minima
% indicara la neurona gnadora, se obtiene el minimo valor y la lista de de
% distancias en cada neurona.
% es importante para saber que donde esta la minima distacia de la señal de
% entrada esta se sutitura para el refinamiento.
[Minval,winlist]=min(dist(W',data')); %data es el vector de entrada
 %W_asoc=W(:,winlist); %pesos asociados a neurona ganadora donde winlist es la BMU posicion
% y W es los pesos asociados a cada neurona
 %Vec_BMU=W_asoc;
end





function net=gg_init(X,netset)
% GG inicializacion 
grd=[0 0;1 0;0 1;1 1]'; %posicion inicial de las 4 neuronas en 
%la base de datos normalizada entre 0 y 1

switch netset.inw
    case 'rn'
        W=rand(size(X,2),size(grd,2)); % el primer elemento 
        %obtiene el número de caracteristicas de columna y 
        %el segundo el numero de neuronas.
        % Se crea un vector de pesos con (2,4)  para las 2 caracteristicas
        %se le asigna el peso de las 4 neuronas.
    case 'mn'
        W=mean(X)'*ones(1,4);
    otherwise
        
end
net.W=W; %se actualiza los pesos de la estructura 
net.grd=grd; % se actualizala posicion de la rejilla.
end
function h=nbrfun(netset,ds,ep)
% Funcion de vecindad puede ser lineal o gaussiana 


%netset.lrvec(ep); %tasa de aprendizaje de la epoca correspondiente


switch netset.neighb
    case 'gauss'
        lr=netset.lrvec(ep); %se escoje la tasa de aprendizaje de la epoca 1
        l = netset.pvec(ep)-ds; %es la distancia entre cada neurona entonces
        %l es la diferencia de la tasa de aprendizaje de epoca i menos distancia de cada neurona 
        l(l<0)=0; %si es menor que 0 se le asigna 0
        l(l>=0)=1; %si es mayor que 0 se le asigna 1
        h = lr.*exp(-(ds.^2)/(2*((netset.pvec(ep))^2))).*l; %esta es la funcion gausiana que calcula h que es la fase de crecimiento
    case 'linear'
        nbr=(1-(ds./(netset.pvec(ep)+1)));
        nbr(nbr<0)=0;
        h = netset.lrvec(ep)*nbr;
end
end



% -------------------------------------------------------------------------
function vishit(net)
%visualizacion del top map
set(gcf,'position',[50 50 640 640],'Color',[1 1 1])
set(gcf, 'name', 'Hit map')
set(gcf, 'NumberTitle', 'off')

maxsize=max(max(net.grd')-min(net.grd'));
gCord=net.grd;

maxbox=450/maxsize;
maxhit=max(net.hitcount);
hold on
plot(gCord(1,:),gCord(2,:),'ws','markersize',maxbox,'markerfacecolor',[1 1 1],'MarkerEdgeColor',[0 0 0])

for i=1:size(net.grd,2)
plot(gCord(1,i),gCord(2,i),'ws','markersize',eps+(net.hitcount(i)/maxhit)*maxbox,'markerfacecolor',[1 0 0],'MarkerEdgeColor',[1 1 1])
% text(gCord(1,i),gCord(2,i),num2str(net.hitcount(i)))
end
axis equal
axis off
hold off
end

function vismap(net,LB)
% Visualizacion de top-map and funcion etiquetado
set(gcf,'position',[50 50 640 640],'Color',[1 1 1])
set(gcf, 'name', 'Top map')
set(gcf, 'NumberTitle', 'off')
% maxsize=max(max(grd'))-min(min(grd'));
maxsize=max(max(net.grd,[],2)-min(net.grd,[],2));
gCord=net.grd;
plot(gCord(1,:),gCord(2,:),'ws','markersize',450/maxsize,'markerfacecolor',[1 1 1],'MarkerEdgeColor',[0 0 0])

axis equal
axis off
text_color = [0 0 0];

for i=1:length(LB)
    text(gCord(1,net.winlist(i))+(0.6*randn/maxsize),gCord(2,net.winlist(i))+(0.6*randn/maxsize),num2str(LB(i)),'horizontalalignment','left','color',text_color)
end
end