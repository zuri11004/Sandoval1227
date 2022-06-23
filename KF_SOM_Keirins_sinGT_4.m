%zurisaddai Sandoval Lara
%Tesis: Self-organizing clustering  by Growing-SOM for EEG Biometrics
%Asesora: Dra. Pilar G�mez Gil
%modificado: 26 de abril 2020.
%Programa que lee las bases de datos de los conjuntos testing y training
%del conjunto con filtro chebychev de orden 2 
%and validation y impostor
%por medio de funciones realiza la verificacion sin refinar las
%caracteristicas solo usando normalizacion de las carecterisiticas
%extraidas por DWT por Db4
%clasifcaci�n por medio de mlp.

%-----------------menu----------------------------------
clear all
clc
while true
disp('opcion [1] : cargar base de datos')
disp('opcion [2] : elaboracion de conjuntos de datos para plantillas por sujeto para mlp')
disp('opcion [3] : Entrenar MLP  ')
disp('opcion [4] : Se crea el conjunto de validacion (validation+impost) with GTrue')
disp('opcion [5] : verificador y metricas  ')
disp('opcion [6] : Graficador de metricas')
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
    %tama�o de las bases de datos
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
             %______________contar cuantas personas hay por sujeto
              [s1,s3,s4,s5,s6] = contar(Mtarget_train);
              vec_valores=horzcat(s1,s3,s4,s5,s6);
              max_con=min(vec_valores); %se obtien el valor maximo 
              %por cada conjunto para crear conjuntos balanceados para el entrenamiento
              %_________________________________
              data_training=[Mtrain_training Mtarget_train];
              %de los conjuntos refinado de training crear 5 
              %conjuntos para los 5 clasificadores que seran las plantillas de cada sujeto
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
                case 3
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
                   neur=input('n�mero de neuronas \n');
                   Epocas=input('n�mero de epocas a entrenar \n');
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
                   
                     case 4
                         %__________se crea la validacion con impostores
                          M_testing=[Mtrain_testing  Mtarget_test] ;
                          M_impost=[Mtrain_impost  Mtarget_impost];
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
                           nombre=cat(2,'Matriz_refinada_KEIRN_testimp_',tarea);
                           fprintf('Nombre del nuevo archivo: %s \n', nombre);
                           save(nombre,'M_prueba');
                         
                      case 5
                          %se cargan la matriz de refinamiento de
                          %validation para pobrar en la plantilla que dice
                          %ser
                          %__________-cambiar nombre por la tarea deseada
                          load('Matriz_refinada_KEIRN_testimp_baseline');
                          
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
                          [Nsuj_aux Nu_car] = size(MRefinada_validation1);              
                           Matriz_excel=zeros(Nsuj_aux,8);
                           umbral=0: 0.05: 1;
                            [f_umbra,c_umbral] = size(umbral);                          % tama�o del umbral para 
                            FRR=zeros([1 c_umbral]);
                            FAR=zeros([1 c_umbral]);%______________________metricas
                        for i_umbral=1:c_umbral
                            um_ele=umbral(1,i_umbral); %se decide con que umbral trabajar
                            TP=0;
                            FP=0;
                            TN=0;
                            FN=0;
                            for i=1:Nsuj_aux
                                   suj_aux= MRefinada_validation1(i,1:Nu_car-2); %vector de entrenamiento con  caracteristicas refinas del conjunto validacion
                                   et_aux= MRefinada_validation1(i,Nu_car-1); % etiqueta auxiliar
                                   gt_identi= MRefinada_validation1(i,Nu_car); %grand true auxiliarya que tenemos la etiqueta reclamamos
                                   %la identidad en la plantilla deseada y permitida y el grand true elegimos al sujeto
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
                                             y = netmlp(z); %se evalua el vector de validacion en la red neuronal                             
                                             Vector_salida = y; %obtiene los valores de la capa de salida
                                             if(Vector_salida(1,1)>=um_ele)
                                                 y_selec=1;
                                             else
                                                  y_selec=0;
                                             end
                                                 
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
                            
                            case 6
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
