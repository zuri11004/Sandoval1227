%zurisaddai Sandoval Lara
%Tesis: Self-organizing clustering  by Growing-SOM for EEG Biometrics
%Asesora: Dra. Pilar Gómez Gil
%modificado: 26 de abril 2020.
%Programa que lee las bases de datos de los conjuntos testing y training
%del conjunto con filtro chebychev de orden 2 y crea una matriz refinada
%por la red som
%and validation y impostor
%por medio de funciones realiza la verificacion  refinando las
%caracteristicas  usando DBGSOM normalizacion de las carecterisiticas
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
    case 2%Entrenamiento de la red DBGSOM
        disp('Entrenamiento de la DBGSOM solo una vez')
        load('pre_param_c4_baseline');
        [~,col]=size(Mtrain_training);
        data1=Mtrain_training;
        class=Mtarget_train;
        %Configura la red neuronal de DBG_SOM 
         %inicializa una estructura predeterminada para los parámetros de la red
         netset.method = 'dbgsom';    % nombre del metodo ('dbgsom')
         netset.mod    = 'batch';   % tipo de aprendizaje: 'batch' or 'sequential'
         netset.epch   = NaN;       % numero de epocas iniciales
         netset.amax   = 0.2;       % max. taza de aprendizaje
         netset.amin   = 0.05;      % min. taza de aprendizaje
         netset.vis    = 'n';       % Visualisacion del entrenamiento (yes, no)
         netset.stnr   =  4;        % numero de neuronas con las que comienza por default 4 (4: square shape , 5:plus shape)
         netset.inw    = 'rn';      % peso inicialpuede ser de eigenvectores o aleatorio (eg: eigenvector, rn: random)
         netset.pmax   = 14.5;         %ancho de la función de vecindad inicial (un número entre 0 y 1) (por defecto = 2)
         netset.pmin   = 0.7;       %ancho de la función de vecindad final (un número entre 0 y pmax)(default = 0.7)
         netset.sf     = NaN;       %factor de dispersión ( 0 > sf > 1) (por defecto = NaN) %
         %_____________________
         pre_param1=pre_param;
         %___________
         disp('numero de epocas para entrenar')
         epoc=input(':');
         netset.epch   = epoc;        % epocas
         netset.sf     = 0.8;        % Factor dispesion
         netset.vis ='n';
        %_________________se termina la configuracion, aqui se usara la la funcion
         %establecida pr el autor de Mahdi Vasighi.
         net=dbgsom(data1,netset,pre_param1);
         
         netDBGsom=net; 
        %_______________guarda red som
         fprintf('Que tarea escogiste? opciones \n letter-composing \n baseline\n');
         fprintf(' multiplication \n rotation \n counting\n');
         tarea = input('escribe entre comillas simples la tarea ');
         nombre1=cat(2,'netDBGsom_',tarea);
         save(nombre1,'netDBGsom');
 
       case 3
           %se refinan los datos de los tres conjuntos
           %se carga la red entrenada
           load('netDBGsom_baseline.mat'); 
          % ____________________________
        MRefinada= RefDBGSOM(Mtrain_training,netDBGsom.W,Mtarget_train);
        MRefinada_training = MRefinada;
        MRefinada= RefDBGSOM(Mtrain_testing,netDBGsom.W,Mtarget_test); %Refina la matriz de prueba
        MRefinada_testing = MRefinada;
        MRefinada= RefDBGSOM(Mtrain_impost,netDBGsom.W,Mtarget_impost);
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
                                  title('Evolution of far and frr with the Threshold')
                                  xlabel('Threshold')
                                  ylabel('FAR and FRR') 
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
function MRefinada= RefDBGSOM(data1,W,class)
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

%_____________________________Funciones para GBSOM
%__________________________________FUNCIONES__________________________________________

function net=dbgsom_init(data)

% DBGSOM initializatio 
grd=[0 0;1 0;0 1;1 1]'; %grid positions
W=mean(data)'*ones(1,4)+0.1*randn(size(data,2),size(grd,2));
        %                 W=rand(size(X,2),size(grd,2));
net.W=W;
net.grd=grd;
end


%_________________
function net=dbgsom(data,netset,pre_param1)

% Training function for DBGSOM 
% Funcion que realizael entremamiento de la som en creciente

[n1,m1]=size(data);
%[data,pre_param]=prefun(data,'rs'); %data pre-processing/Pre-proceso de datos %normalizacion, eliminar ouesto que nuestra matriz de datos se encuentra normalizda
data_tmp=data;
pre_param=pre_param1;

net=dbgsom_init(data); %gsom initialization /llama a la funcion de inicializacion de la red

ds=linkdist(net.grd); %es una función de distancia de capa utilizada para encontrar las distancias entre las neuronas de la capa dadas sus posiciones.

epvec=1:netset.epch; %epocas contador vector

netset.pvec=(netset.pmax-netset.pmin)*((netset.epch-epvec)/(netset.epch-1))+netset.pmin;

netset.lrvec=(netset.amax-netset.amin)*((netset.epch-epvec)/(netset.epch-1))+netset.amin; % Taza de aprendizaje
netset.GT=-m1*log(netset.sf);

fprintf('Data matrix (Samples=%i Variables=%i)\n',n1,m1)
fprintf('DBGSOM (SF=%3.2f)\n',netset.sf)

for ep=1:netset.epch %contaor de epocas
    if ~rem(ep,10) %funcion que saca el resto
       fprintf('Training epoch %i\n',ep)% Imprimir los divisibles
    end
        
    h=nbrfun(netset,ds,ep); %Funcion de vecindad(que se utiliza)
    [~,net.win_list]=winfun(data,net.W); %neurona ganora
    %__________checar bien como se realiza la actualizacion de nuronas fase
    %de ccrecimiento
    hh=h(:,net.win_list);
    h1=data'*hh';
    h2= true(size(data'))*hh';
    pos = find(h2 > 0);
    net.W(pos)=h1(pos)./h2(pos); %acutalizacion de neuronas
    %________________________________
    if ep<netset.epch && ep>1
    
    %calulcating error of each neuron
    %calcular el error en cada neurona
    [win_err,win_id]=winfun(data,net.W); %con la misma funcion de distancia de neuronas 
    net.Err=errcalc(win_err,win_id,net.grd); %funcion que calcula el error
    
    % check to find boundary and non-boundary nodes.
    %encontrar los nodos mas cercanos 
    [nb_id,bo_id]=findbound(ds); %funcion para encontrar los nodos mas cercanos 
    
    % distibute the error of non-boundary nodes which has a higher error
    
    %distribuye el error de los nodos no fronterizo . que tienen un alto
    %error
    %_______________________modelo de 
    % than GT (batch mode) %modo por lotes
    for jj=1:length(nb_id)
        if net.Err(nb_id(jj))>=netset.GT
            intersect_id=intersect(find(ds(nb_id(jj),:)==1),bo_id);
            if ~isempty(intersect_id)
                net.Err(intersect_id)=net.Err(intersect_id)+net.Err(nb_id(jj))/8; % error of a nb node is distributed equally between neighbors and itself
                net.Err(nb_id(jj))=net.Err(nb_id(jj))/2;
            end
        end
    end
    %___________________________________________
    net=dbgrowfun(net,netset,ds,bo_id); % funcion de crecimiento esta parte realiza la mayor parte. %la fase de crecimiento de la red
    
    ds=linkdist(net.grd);
    %____________________
    end%_____termina el if
    % ploting neurons in data sapce
    %inserta las neuronas
    if strcmp(netset.vis,'y')
        vistrn(net,data)
    end    
    
end
fprintf('[Done]\n')

[~,winlist]=winfun(data_tmp,net.W);
[hitcount,~]=hist(winlist,size(net.W,2));
net.winlist=winlist;
net.hitcount=hitcount;
net.pre_param=pre_param; 
%net=net_eval(data,net); no se hace la evaluacion

end

%__________________ funcion que normaliza la base de datos
function [pre_X,pre_param]=prefun(X,option)
% Preprocessing function %funcion de preprocesamiento de los datos 
% apply range-scaling or function [pre_X,pre_param]=prefun(X,option)
% Preprocessing function 
% apply range-scaling or auto-scaling

%determinar el tamaño de datos  

[n1,~]=size(X);

switch option
    case 'rs' % range scaling the columns
        pre_X=(X-repmat(min(X),n1,1))./(repmat(max(X)-min(X),n1,1));
        pre_param.min=min(X);
        pre_param.max=max(X);
    case 'as' % mean centering
        pre_X=(X-repmat(mean(X),n1,1))./(repmat(std(X),n1,1));
        pre_param.cmean=mean(X);
        pre_param.cstd=std(X);
end
end

%___________________________ FUNCION de vencindad
function h=nbrfun(netset,ds,ep)

% neighbor function /funcion de vecindad
lr=netset.lrvec(ep); %taza de aprendizaje

% l = netset.pvec(ep)-ds;
% l(l<0)=0;
% l(l>=0)=1;
% h = lr.*exp(-(ds.^2)/(2*netset.pvec(ep)^2)).*l; %checar esta funcion

nbr=(1-(ds./(netset.pvec(ep)+1)));
nbr(nbr<0)=0;
h = lr.*nbr;
end
%_______________Funcion de neurona ganadora
function [Minval,winlist]=winfun(data,W)
% finding list of winner neurons and corresponding error values for data 
%encuentra la lista de neuronas ganadoras y valores correspondiente de
%errores por los datos


[Minval,winlist]=min(dist(W',data'));   %saca la mini distancia y en la lista de neuronas y su error al valor
end

%____________funcion de error acumulativo
function Err=errcalc(win_err,win_id,grd)
% Accumulative error calculation
%calculo error acumulativo

Err=zeros(1,size(grd,2));
for i=1:size(grd,2)
    Err(i)=sum(win_err(find(win_id==i)));
end
end

%______________-funcin de encontrar las neuronas proximas fronterizas y no
%fornterizas
function [nb_id,bo_id]=findbound(ds)
% finding boundary and non-boundary neurons

nb_id=find(sum(ds==1)==4); % nodos no fronterizo
bo_id=find(sum(ds==1)<4);  % nodos fronterizos
end

%_____________funcion de crecieminto_______________parte importante
function net=dbgrowfun(net,netset,ds,bo_id)
% Growing phase function in batch mode learning
%fase de crecimiento en modo aprendizje por  lote 


W=net.W;
Err=net.Err;
grd=net.grd;


[sorted_Err,x1id]=sort(Err(bo_id),'descend');
bo_id=bo_id(x1id);
bo_id=bo_id(Err(bo_id)>=netset.GT); % Run growing phase only for b-node (Err>=GT)

%Growing phase
for i=1:length(bo_id)
    new_grd=[];
    W_new_grd=[];
    new_Err=[];
    
    notallowed_pos=[ismember([grd(:,bo_id(i))+[0;1]]',grd','rows') ismember([grd(:,bo_id(i))+[0;-1]]',grd','rows'),...
        [ismember([grd(:,bo_id(i))+[1;0]]',grd','rows') ismember([grd(:,bo_id(i))+[-1;0]]',grd','rows')]];
    
    direct_opr=[0 0 1 -1;1 -1 0 0];%[N S E W] directions
    direct_opr(:,notallowed_pos)=[];
    allowed_grd=repmat(grd(:,bo_id(i)),1,size(direct_opr,2))+direct_opr;
    
    
    if ~isempty(allowed_grd)

        nbr1_id=find(ds(bo_id(i),:)==1);
        nbr2_id=find(ds(bo_id(i),:)==2);
    
        %sorting neighbirs of boundary nodes
                [sorted_Err1,x1id]=sort(Err(nbr1_id),'descend');%
                nbr1_id=nbr1_id(x1id);
                [sorted_Err2,x2id]=sort(Err(nbr2_id),'descend');%
                nbr2_id=nbr2_id(x2id);
    
    
    if size(allowed_grd,2)==3 && length(nbr1_id)==1 %three possible postion to grow
        sel_nbr2_id = nbr2_id(min(dist(allowed_grd',grd(:,nbr2_id)))==1);
        nbr1_dist = dist(grd(:,nbr1_id)',allowed_grd);
        new_Err=0;
        %Actualizacion de pesos
        if isempty(sel_nbr2_id) % if there is no allowed nbr2
            temp_id=find(nbr1_dist~=2);
            new_grd = allowed_grd(:,temp_id(1));
            W_new_grd = 2.*W(:,bo_id(i))-W(:,nbr1_id);
            
        else % there is nbr2
            [mval,max_id]=max(Err(sel_nbr2_id));
            nbr2_dist = dist(allowed_grd',grd(:,sel_nbr2_id(max_id)));
            new_grd = allowed_grd(:,nbr2_dist==1);
            
            W_new_grd = (W(:,bo_id(i))+W(:,sel_nbr2_id(max_id)))/2;
        end
        
    
    elseif size(allowed_grd,2)==2 && length(nbr1_id)==2%two possible positions to grow
        [~,max_id]=max(Err(nbr1_id));
        [~,min_id]=min(Err(nbr1_id));
        [~,ind]=min(dist(grd(:,nbr1_id(max_id))',allowed_grd));
        
        sel_nbr2_id = nbr2_id(min(dist(allowed_grd',grd(:,nbr2_id)))==1);
        [~,max2_id]=max(Err(sel_nbr2_id));
        
        new_grd=allowed_grd(:,ind(1));
        new_Err=0;
        
        if isempty(sel_nbr2_id) % if there is no adj nbr2
            W_new_grd = 2.*W(:,bo_id(i))-W(:,nbr1_id(min_id)); % can be change (mean with bo)
        else
            if dist(grd(nbr1_id(max_id))',grd(:,sel_nbr2_id(max2_id)))==1 % nbr2 is adj to selected allowed grd
                W_new_grd = (2.*W(:,bo_id(i))-W(:,nbr1_id(min_id))+W(:,sel_nbr2_id(max2_id)))/2;
            else
                W_new_grd = 2.*W(:,bo_id(i))-W(:,nbr1_id(min_id));
            end
        end
        
  
    
    else
        if length(nbr1_id)==3 %one possible position to grow
            new_grd=allowed_grd;
            new_Err=0;
            nbr1_dist = dist(allowed_grd',grd(:,nbr1_id));
            sel_nbr2_id = nbr2_id(min(dist(allowed_grd',grd(:,nbr2_id)))==1);
            
            if isempty(sel_nbr2_id) % if there is no adj nbr2
                W_new_grd = 2.*W(:,bo_id(i))-W(:,nbr1_id(nbr1_dist==2));
            else
                [~,max2_id]=max(Err(sel_nbr2_id));
                W_new_grd = (2.*W(:,bo_id(i))-W(:,nbr1_id(nbr1_dist==2))+W(:,sel_nbr2_id(max2_id)))/2;
            end
            
        end
    end
    grd=[grd,new_grd];
    W=[W,W_new_grd]; %actualizar pesos
    Err=[Err,new_Err];
    
    end
end
net.grd=grd;
net.W=W;
end

%________funcion_sin ocupar para distorision de mapa para las evaluacion de
%los datos de entrenamiento
function net=net_eval(data,net)
% Evaluation of the trained SOM network
% QE: Quantization error
% TE: Topographic error
% DB: Davies–Bouldin index
% DM: Distorsion measure

net.eval.QE=qefun(data,net);
net.eval.TE=tefun(data,net);
net.eval.DB=dbfun(data,net);
net.eval.DM=dmfun(data,net);
end

function DM=dmfun(data,net)
% Calculate distorsion measure

W=net.W;
winlist=net.winlist;
hitcount=net.hitcount;

[data,~]=prefun(data,'rs');

ntds=linkdist(net.grd);
h = exp(-(ntds.^2)/(2*((1)^2)));
hh=h(:,winlist);
dst=dist(data,W);

hitind=hitcount>0;
dm=hh.*(dst.^2)';
dm=dm(hitind,:);
DM=mean(sum(dm./repmat(hitcount(hitind)',1,size(data,1))));
end

function DB=dbfun(data,net)
% Calculate Daviesâ€“Bouldin index

W=net.W;
hitcount=net.hitcount;
winlist=net.winlist;

[data,~]=prefun(data,'rs');

for i=1:size(W,2)
    for j=i+1:size(W,2)
        if hitcount(j)~=0 && hitcount(i)~=0;
        di=mean(dist(data(winlist==i,:),W(:,i)));
        dj=mean(dist(data(winlist==j,:),W(:,j)));
        dij=dist(W(:,i)',W(:,j));
        D2(i,j)=(di+dj)/dij;
        D2(j,i)=(di+dj)/dij;
        end
    end
end
DB=sum(max(D2))/sum(hitcount~=0);
end
function QE=qefun(data,net)
% Calculate quantization error of the SOM


W=net.W;
[data,~]=prefun(data,'rs');
[sor_val,~]=sort(dist(W',data'));
QE = sum(sor_val(1,:)) / length(sor_val(1,:));
end
function TE=tefun(data,net)
% Calculate Topographic Error


W=net.W;

[data,~]=prefun(data,'rs');

[~,sor_ind]=sort(dist(W',data'));
bmu1_ind=sor_ind(1,:);
bmu2_ind=sor_ind(2,:);

Da=linkdist(net.grd);

for i=1:length(bmu1_ind)
    isadj(i)=Da(bmu1_ind(i),bmu2_ind(i));
end
TE =sum(isadj~=1)/length(isadj);
end


%_____________visualizacion del mapa funciones
function vistrn(net,data)
% training visualization function
%Entrenamiento visual

[xcord,ycord]=find(triu(linkdist(net.grd))==1);
plot(data(:,1),data(:,2),'.','markersize',14)
hold on
plot(net.W(1,:),net.W(2,:),'.r','markersize',18,'markerfacecolor','r')
for ii=1:length(xcord)
    line([net.W(1,xcord(ii)),net.W(1,ycord(ii))],[net.W(2,xcord(ii)),net.W(2,ycord(ii))],'Color','r');
end
axis equal
% axis off
xlim([-0.5 1.5])
ylim([-0.5 1.5])
hold off
drawnow
end
function vis_lab(LB,top_id,grd)
% top map visualization function


set(gcf,'position',[20 40 640 640],'Color',[1 1 1])
% maxsize=max(max(grd'))-min(min(grd'));
maxsize=max(max(grd')-min(grd'));
gCord=grd;
plot(gCord(1,:),gCord(2,:),'ws','markersize',450/maxsize,'markerfacecolor',[1 1 1],'MarkerEdgeColor',[0 0 0])

axis equal
axis off
text_color = [0 0 0];
while ~isempty(LB)
    ww=find(top_id==top_id(1));
    for i=1:length(ww)
        text(gCord(1,top_id(1))+(0.7*randn/maxsize),gCord(2,top_id(1))+(0.7*randn/maxsize),num2str(LB(ww(i))),'fontsize',150/maxsize,'horizontalalignment','left','color',text_color)
    end
    top_id(ww)=[];
    LB(ww)=[];
end
end

function vishit(net)
% hit map visualization function
set(gcf,'position',[50 50 640 640],'Color',[1 1 1])
set(gcf, 'name', 'Hit map')
set(gcf, 'NumberTitle', 'off')
% maxsize=max(max(grd'))-min(min(grd'));
maxsize=max(max(net.grd')-min(net.grd'));
gCord=net.grd;

maxbox=450/maxsize;
maxhit=max(net.hitcount);
hold on
plot(gCord(1,:),gCord(2,:),'ws','markersize',maxbox,'markerfacecolor',[1 1 1],'MarkerEdgeColor',[0 0 0],'LineWidth',1.5)

for i=1:size(net.grd,2)
plot(gCord(1,i),gCord(2,i),'ws','markersize',eps+(net.hitcount(i)/maxhit)*maxbox,'markerfacecolor',[1 0 0],'MarkerEdgeColor',[1 1 1])
% text(gCord(1,i),gCord(2,i),num2str(net.hitcount(i)))
end
axis equal
axis off
hold off
end

function vislay(net,laynum)
% weight map (weight layer) visualization function

set(gcf,'position',[50 50 640 640],'Color',[1 1 1])
set(gcf, 'name', ['Weight layer',num2str(laynum)])
set(gcf, 'NumberTitle', 'off')

% maxsize=max(max(grd'))-min(min(grd'));
maxsize=max(max(net.grd')-min(net.grd'));
gCord=net.grd;

maxbox=430/maxsize;
maxval=max(net.W(laynum,:));
hold on
% plot(gCord(1,:),gCord(2,:),'ws','markersize',maxbox,'markerfacecolor',[1 1 1],'MarkerEdgeColor',[0 0 0],'LineWidth',1.5)

for i=1:size(net.grd,2)
plot(gCord(1,i),gCord(2,i),'ws','markersize',maxbox,'markerfacecolor',(net.W(laynum,i)/maxval).*[1 1 1],'MarkerEdgeColor',[0 0 0],'LineWidth',1.5)
% text(gCord(1,i),gCord(2,i),num2str(net.hitcount(i)))
end
axis equal
axis off
hold off
end
function vismap(net,LB)
% Top-map visualization function for SOM

set(gcf,'position',[50 50 640 640],'Color',[1 1 1])
set(gcf, 'name', 'Top map')
set(gcf, 'NumberTitle', 'off')
% maxsize=max(max(grd'))-min(min(grd'));
maxsize=max(max(net.grd,[],2)-min(net.grd,[],2));
gCord=net.grd;
plot(gCord(1,:),gCord(2,:),'ws','markersize',450/maxsize,'markerfacecolor',[1 1 1],'MarkerEdgeColor',[0 0 0],'LineWidth',1.5)

axis equal
axis off
text_color = [0 0 0];
while ~isempty(LB)
    ww=find(net.winlist==net.winlist(1));
    for i=1:length(ww)
%         text(gCord(1,net.winlist(1))+(0.7*randn/maxsize),gCord(2,net.winlist(1))+(0.7*randn/maxsize),num2str(LB(ww(i))),'fontsize',150/maxsize,'horizontalalignment','left','color',text_color)
        text(gCord(1,net.winlist(1))+(0.6*randn/maxsize),gCord(2,net.winlist(1))+(0.6*randn/maxsize),num2str(LB(ww(i))),'horizontalalignment','left','color',text_color)
    end
    net.winlist(ww)=[];
    LB(ww)=[];
end

end