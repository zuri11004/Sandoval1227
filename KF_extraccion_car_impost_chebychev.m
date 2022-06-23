%zurisaddai Sandoval Lara
%Tesis: Self-organizing clustering  by Growing-SOM for EEG Biometrics
%Asesora: Dra. Pilar Gómez Gil
%cCCreado: 25 de Abril del 2019.
%base de datos de keirins con caracteristicas normalizadas.

%usando los conjuntos de datos obtenidos por validacion cruzada k-folds 10
%veces y la TDW que tiene complejidad O(N) ademas con db4 que tuvo buenos
%resultados para las señales de EEG y competitivos como MODTW
%la funcion  Fbandas realizada inspirada por Edgar Hernandez Gonzalez  
%Proceso de pasabajas a 64 hz, para  bajar la señal.
clear all 
clc
 % Diseño del Filtro del filtro pasabajass
    Fs=250;  %Frecuencia de muestreo 250 Hz de la señal
    N=2500; % numero de muestras 
    T=1/Fs; 
    Wp=64/(Fs/2); % Banda de Paso (Frecuencia de corte normalizada)
    Rp=3;         % Rizo en la frecuencia de corte
    Ws=65/(Fs/2); % Frecuencia en banda de rechazo %revisar eso parte de porque 65
    Rs=60;        % Atenuación en la banda de rechazo de acuerdo a la  tesis de ever para filtro cheby2ord
    %_______________________________se obtiene el un filtro pasa bajas de
    %chebychev tipo 2
    [nx,Wnx]=cheb2ord(Wp,Ws,Rp,Rs);
    [b,a]=cheby2(nx,Rs,Wnx, 'low'); %Filtro Cheb tipo 2 
    
     %_______________Se carga la base de datos de entrenamiento

       load('KEIRN_impost_rotation'); % se carga la matriz original con sus etiquetas 
       %load('KEIRN_testingc1_letter_composing');
       %load('KEIRN_testingc1_multiplication'); % se carga la matriz original con sus etiquetas
       %load('KEIRN_testingc1_baseline');
       %load('KEIRN_testingc1_counting');
     %____________hasta aqui
       load('pre_param_rotation');%se carga la estructura con valores min y max para normalizr
       %load('pre_param_dia_counting');
      canal=input('numero de canales de las bases de datos : ');
      T_caract=(4*3*(canal))+ 1;%4 caract por 3 bandas por 6 canales + 1 etiqueta
      Matriz_carac_f=zeros([1 T_caract]); %MMatriz de caracteristicas
      recortado=impostor; %Estructura de la base de datos de conjunto i para el testing
      % recortado=impostor;
      [~,c] = size(recortado);
for k=1:c % for para recorrer las celdas de todo el conjunto 
      matriz_datos= recortado{1,k}{1,4}; %datos de un solo sujeto
      etiqueta=recortado{1,k}{1,1};
      for i=1:6  % para obtener las bandas de los 6 canales y sacar la transformada wavelet
           
%            if(i==1)
%            canal_aux=matriz_datos(i,:); %canal c3
%            %Este filtrado deja solo la señal por debajo de 64 Hz        
%            % FILTRO PASA BAJAS con valores de chevycheb tipo 2;
%            canal_aux=double(canal_aux);
%            canal_pb = filtfilt(b, a, canal_aux);
%            Vector_final= FBandas(canal_pb,'db4'); %calcula las bandas de frecuencia por canal usando como wavelet madre daubechis de 4 niveles 
%            Vector_final1= Vector_final;
%            end
         
          if(i==2)
            canal_aux=matriz_datos(i,:); %canal c4
            canal_aux=double(canal_aux);
            canal_pb = filtfilt(b, a, canal_aux);
            Vector_final= FBandas(canal_pb,'db4'); %calcula las bandas de frecuencia por canal
            Vector_final2= Vector_final;
           end
%            if(i==3)
%             canal_aux=matriz_datos(i,:); %canal p3
%             canal_aux=double(canal_aux);
%             canal_pb = filtfilt(b, a, canal_aux);
%             Vector_final= FBandas(canal_pb,'db4'); %calcula las bandas de frecuencia por canal
%             Vector_final3= Vector_final;
%            end
%            if(i==4)
%             canal_aux=matriz_datos(i,:); %canal p4
%             canal_aux=double(canal_aux);
%             canal_pb = filtfilt(b, a, canal_aux);
%             Vector_final= FBandas(canal_pb,'db4'); %calcula las bandas de frecuencia por canal
%             Vector_final4= Vector_final;
%            end
           if(i==5)
           canal_aux=matriz_datos(i,:); %canal o1
           canal_aux=double(canal_aux);
           canal_pb = filtfilt(b, a, canal_aux); % canal pasado por un filtro pasa bajas con filtro chebychev orden 2
           Vector_final= FBandas(canal_pb,'db4'); %calcula las bandas de frecuencia por canal
           Vector_final5= Vector_final;
           end
%            if(i==6)
%            canal_aux=matriz_datos(i,:); %canalo2
%            canal_aux=double(canal_aux);
%            canal_pb = filtfilt(b, a, canal_aux); % canal pasado por un filtro pasa bajas
%            Vector_final= FBandas(canal_pb,'db4'); %calcula las bandas de frecuencia por canal
%            Vector_final6= Vector_final;
%            end
      end
      
       
            target=targt(etiqueta); %clase seleccionada
            Vector_carac_total=horzcat(Vector_final2,Vector_final5,target);
            %Vector_carac_total=horzcat(Vector_final1,Vector_final2,Vector_final3,Vector_final4,Vector_final5,Vector_final6,target); %vector por sujeto con los 6 canales y la etiqueta.
            Matriz_carac_f = [Matriz_carac_f;Vector_carac_total]; %se va agregando a la matriz caracteritica losvectores de caracteristicas
end
     [tam_f,~]=size(Matriz_carac_f);
     Matriz_carac_final = Matriz_carac_f(2:tam_f, :);  %se tiene la matriz de caracteristicas finales
     s_noma=pre_param;
     %Se normalizara los datos
     [data]=Normal(Matriz_carac_final,s_noma);
     
  fprintf('Tareas: \n letter-composing \n baseline\n');
  fprintf(' multiplication \n rotation \n counting\n');
  tarea = input('escribe ci tarea');
  nombre2=cat(2,'Matriz_norm_keirns_impost_',tarea);
  %nombre2=cat(2,'Matriz_norm_keirns_impost_dia_',tarea);
  save(nombre2,'data');
   
      
   %___________________Se  guarda el archivo de la matriz normal y la
   %estructura de vectores min y max

     
     
     
     
     
     
     %______________________________________________________Funcionesn_______________________
 function [pre_X]=Normal(X,pre_param) %se debe cancelar
%normaliza los datoss
 [~,m1]=size(X); %tamañano de lasfilas nada mas, anula columnas
 class=X(:,m1);
 X=X(:,1:(m1-1));
 [n1,m1]=size(X);
   col_min= pre_param.min;
   col_max=pre_param.max;
   pre_X=X;
        %______________________
      for i1=1:m1 %ultima columna sin target
         col_aux=X(:,i1);
        % se normalizan los valores obtiene su maximo y su minimo
%                   % de todos los valores
           maximo=col_max(1,i1); %se tiene una columna de los valores maximos del training
           
           minimo=col_min(1,i1); %%se tiene una columna de los valores minimos del training
           resta=maximo - minimo;
           if(resta==0)
                 for i2=1:n1
                     col_aux(i2,1)= ((col_aux(i2,1) - (minimo))/ (resta));
                     col_aux(i2,1)=0;
                 end %se normaliza entre  0 y 1 los valores  y se normaliza con los valores fijos del training.
                 
           else
                 for i2=1:n1
                     col_aux(i2,1)= ((col_aux(i2,1) - (minimo))/ (resta));
                 end %se normaliza entre  0 y 1 los valores  y se normaliza con los valores fijos del training.
           end
        pre_X(:,i1)=col_aux;      
      end
        %______________________
   
        %pre_X=(X-repmat(min(col_min),n1,1))./(repmat(max(col_max)-min(col_min),n1,1));

   
        pre_X=horzcat(pre_X,class);
end
       function targ=targt(a)   
       %Funcion que regresa la clase del seleccionado
       if (a == "subject 1" )
            targ=1;
        end
        if (a == "subject 2" )
            targ=2;
        end
        if (a == "subject 3" )
            targ=3;
        end
        if (a == "subject 4" )
            targ=4;
        end
        if (a == "subject 5" )
            targ=5;
        end
        if (a == "subject 6" )
            targ=6;
        end
        if (a == "subject 7" )
            targ=7;
        end
       end
       
       function Vector_final=FBandas(x,wavelet)
       %funcion que regresa el vector caracteriztico de estadistica con una
       %db4 para el vector de entrada del canal seleccionado
       %Descomposicion Wavelet multinivel 1-D (DWT) en 4 niveles
        [C,L] = wavedec(x,4,wavelet);
       %Coeficiente de detalle 3 (8Hz-16Hz), de aqui sacar alfa
        cD3=detcoef(C,L,3);
        cD2 = detcoef (C,L,2); % coef de 16-32 hz
        % Transformada Wavelet Discreta (DWT) 1-D de un solo nivel a cD3 para obtener alfa
        [cA,cD]=dwt(cD3,wavelet); %Descomposicion 1-D (cA=alfa)
         %---------------------------------------------------------------------------------
        delta=appcoef(C,L,wavelet,4); %Coeficiente de aproximacion 4(0Hz-4Hz)
         %calcula valores estadisticos de cada banda media, varianza,
         %mediana absoluta
         %Aseguramos que sea Delta
                 banda=delta;
                 Media_delta= mean(banda);
                 Mediana_delta= abs(median(banda));
                 Varianza_delta= var(banda,0);
                 Entropia_delta= wentropy(banda,'shannon');
                 
      %-----------------------------------------------------------------
        theta=detcoef(C,L,4); %Coeficiente de detalle 4(4Hz-8Hz)
                 banda = theta;
                 Media_theta= mean(banda);
                 Mediana_theta= abs(median(banda));
                 Varianza_theta= var(banda,0);
                 Entropia_theta= wentropy(banda,'shannon');
    %_______________________________________________________________
        alfa=cA; %8Hz-12Hz
                 banda = alfa;
                 Media_alfa= mean(banda);
                 Mediana_alfa= abs(median(banda));
                 Varianza_alfa= var(banda,0);
                 Entropia_alfa= wentropy(banda,'shannon');
                 
    % Se crea el vector final con las características de cada sub-banda
        VectorFinal=[];
        
        for x1=1:3
            
            if x1==1 %Aseguramos que sea Delta
                 VectorFinal = [VectorFinal, Media_delta];
                 VectorFinal = [VectorFinal, Mediana_delta];
                 VectorFinal = [VectorFinal, Varianza_delta];
                 VectorFinal = [VectorFinal, Entropia_delta];
            elseif x1==2 %Aseguramos que sea Theta
                 VectorFinal = [VectorFinal, Media_theta];
                 VectorFinal = [VectorFinal, Mediana_theta];
                 VectorFinal = [VectorFinal, Varianza_theta];
                 VectorFinal = [VectorFinal, Entropia_theta];
            elseif x1==3 % Esto es Alpha
                 VectorFinal = [VectorFinal, Media_alfa];
                 VectorFinal = [VectorFinal, Mediana_alfa];
                 VectorFinal = [VectorFinal, Varianza_alfa];
                 VectorFinal = [VectorFinal, Entropia_alfa];
            end
        end
        Vector_final= VectorFinal;
       end