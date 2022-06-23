% programa para leer la base de datos KEIRN y generar otro archivo para una
% tarea específica
% Realizado por Pilar Gómez Gil, 25 de Noviembre 2019

clear variables;

% leyendo la tarea deseada.
fprintf('Que tarea deseas? opciones \n letter-composing \n baseline\n');
fprintf(' multiplication \n rotation \n counting\n');
tarea = input('... escribe entre comillas simples la tarea que quieres ');

% Carga los datos
load('eegdata.mat');
[m,n]=size(data);
fprintf(' renglones = %d   columas = %d \n',m,n);
escogidos = cell(m,n);
i_escogido = 1;
% recorriendo todas las celdas
for i=1:1:n
    % fprintf('accesando la celda %d \n',i);
    estaCelda=data(1,i);
    todos = estaCelda{1,1};
    esteSubject = todos{1,1};
    esteTask = todos{1,2};
    esteTrial = todos{1,3};
    % disp(esteSubject);
    % disp(esteTask);
    % disp(esteTrial);
    if strcmp(esteTask,tarea)
        % fprintf(' es la tarea que quieres!\n');
        escogidos{1,i_escogido}=data{1,i};
        i_escogido = i_escogido+1;
    end
    % pause;
end
fprintf('\n Se seleccionaron %d elementos\n',i_escogido-1);
recortado = escogidos(1:(i_escogido-1));
nombre=cat(2,'KEIRN_task_',tarea);
fprintf('Nombre del nuevo archivo: %s \n', nombre);
save(nombre,'recortado');





    
