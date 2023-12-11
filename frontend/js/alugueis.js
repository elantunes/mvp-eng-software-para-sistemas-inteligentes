SERVIDOR_API = 'http://localhost:5001/';

btnNovoAluguel.onclick = () => ativar_newForm();


btnSalvarAluguel.onclick = function() {
  const dados = ler_aluguelForm();
  if (valida_dados_insercao_aluguel(dados))
    post_aluguel(dados);
};


btnSalvarAlteracoesAluguel.onclick = function() {
  const dados = ler_aluguelForm();
  if (valida_dados_insercao_aluguel(dados))
    put_aluguel(dados);
};


dateDataInicio.onchange = function(){
  calcular_valor_total_aluguel();
};


dateDataTermino.onchange = function(){
  calcular_valor_total_aluguel();
};


selModeloVeiculo.onchange = function(){
  calcular_valor_total_aluguel();
};


txtCpfCliente.onkeyup = function(event) {
  const cpf = event.target.value;
  if (cpf.length == 11)
    get_cliente(cpf);
  else
    lblNomeCliente.innerHTML = '&nbsp';
}


/*
  --------------------------------------------------------------------------------------
  Deleta um aluguel via requisição DELETE
  --------------------------------------------------------------------------------------
*/
const delete_aluguel = async (idAluguel) => {
  const url = `${SERVIDOR_API}aluguel/${idAluguel}`;
  fetch(url, {
    method: 'delete',
  })
  .then((response) => response.json())
  .then(function() {
    closeModal(divModalDeleteAluguel);
    get_alugueis();
    ativar_popup_operacao_com_sucesso();
  }).catch((error) => console.error('Error:', error));
}


/*
  --------------------------------------------------------------------------------------
  Obtém uma lista de aluguéis via requisição GET
  --------------------------------------------------------------------------------------
*/
const get_alugueis = async () => {
  const url = `${SERVIDOR_API}alugueis`;
  fetch(url, {
    method: 'get',
  })
  .then((response) => response.json())
  .then(function(data) {
    render_alugueis(data.alugueis);
  })
  .catch((error) => console.error('Error:', error));
}


/*
  --------------------------------------------------------------------------------------
  Obtém um aluguel via requisição GET
  --------------------------------------------------------------------------------------
*/
const get_aluguel = async (id) => {
  const url = `${SERVIDOR_API}aluguel/${id}`;
  fetch(url, {
    method: 'get',
  })
  .then((response) => response.json())
  .then((data) => ativar_editForm(data))
  .catch((error) => console.error('Error:', error));
}


/*
  --------------------------------------------------------------------------------------
  Obtém as informações de um CLiente pelo CPF
  --------------------------------------------------------------------------------------
*/
const get_cliente = async(cpf) => {
  const url = `${SERVIDOR_API}cliente`;

  const formData = new FormData();
  formData.append('cpf', cpf);

  fetch(url, {
    method: 'get',
    headers: { 'cpf':cpf }
  })
  .then((response) => response.json())
  .then(function(data) {
    hidIdCliente.value = !data.hasOwnProperty('erro') ? data.id : '';
    lblNomeCliente.innerHTML = !data.hasOwnProperty('erro') ? data.nome : 'CPF não encontrado';
  })
  .catch((error) => console.error('Error:', error));
}


/*
  --------------------------------------------------------------------------------------
  Obtém uma lista de modelos veículos via requisição GET
  --------------------------------------------------------------------------------------
*/
const get_modelos_veiculos = async () => {
  const url = `${SERVIDOR_API}veiculos`;
  fetch(url, {
    method: 'get',
  })
  .then((response) => response.json())
  .then(function(data) {
    for(var veiculo of data.veiculos)
      selModeloVeiculo.innerHTML += `<option value='${veiculo.id_veiculo}' data-valordiaria='${veiculo.valor_diaria}'>${veiculo.modelo}</option>`;
  })
  .catch((error) => console.error('Error:', error));
}


/*
  --------------------------------------------------------------------------------------
  Insere um aluguel via requisição POST
  --------------------------------------------------------------------------------------
*/
const post_aluguel = async (aluguel) => {
  const url = `${SERVIDOR_API}aluguel`;

  const formData = new FormData();
  formData.append('id_cliente', aluguel.id_cliente);
  formData.append('id_veiculo', aluguel.id_veiculo);
  formData.append('data_inicio', aluguel.data_inicio + ' 00:00:00');
  formData.append('data_termino', aluguel.data_termino + ' 00:00:00');

  fetch(url, {
    method: 'post',
    body: formData
  })
  .then((response) => response.json())
  .then(() => {
      closeModal(divModalAluguel);
      get_alugueis();
      ativar_popup_operacao_com_sucesso();
  })
  .catch((error) => console.error('Error:', error));
}


/*
  --------------------------------------------------------------------------------------
  Altera um aluguel via requisição PUT
  --------------------------------------------------------------------------------------
*/
const put_aluguel = async (aluguel) => {
  const url = `${SERVIDOR_API}aluguel/${aluguel.id}`;

  const formData = new FormData();
  formData.append('id_veiculo', aluguel.id_veiculo);
  formData.append('data_inicio', aluguel.data_inicio + ' 00:00:00');
  formData.append('data_termino', aluguel.data_termino + ' 00:00:00');

  delete aluguel.id;

  fetch(url, {
    method: 'put',
    body: formData
  })
  .then((response) => response.json())
  .then(() => {
    closeModal(divModalAluguel);
    get_alugueis();
    ativar_popup_operacao_com_sucesso();
  })
  .catch((error) => console.error('Error:', error));
}


function ativar_confirmacao_delecao(idAluguel) {
  hiddenIdAluguelModalDeleteAluguel.value = idAluguel;
  divModalDeleteAluguel.style.display = 'flex';
}


function ativar_editForm(aluguel) {
  titulo.innerHTML = 'Editar Aluguel';

  const data_inicio = string_para_date(aluguel.data_inicio);
  const data_termino = string_para_date(aluguel.data_termino);

  hiddenId.value = aluguel.id;
  txtCpfCliente.value = aluguel.cliente.cpf;
  txtCpfCliente.disabled = 'disabled';
  lblNomeCliente.innerHTML = aluguel.cliente.nome;
  selModeloVeiculo.value = aluguel.veiculo.id;
  dateDataInicio.valueAsDate = data_inicio;
  dateDataTermino.valueAsDate = data_termino;

  render_infos_aluguel(aluguel.veiculo.valor_diaria, calcular_numero_dias(data_inicio, data_termino));

  btnSalvarAluguel.style.display = 'none';
  btnSalvarAlteracoesAluguel.style.display = 'block';

  divModalAluguel.style.display = 'flex';
}


function ativar_newForm() {
  titulo.innerHTML = 'Novo Aluguel';

  const data_inicio = new Date();
  const data_termino = new Date();

  txtCpfCliente.value = '';
  txtCpfCliente.disabled = '';
  lblNomeCliente.innerHTML = '&nbsp;';
  selModeloVeiculo.value = '';
  dateDataInicio.valueAsDate = data_inicio;
  dateDataTermino.valueAsDate = data_termino;

  render_infos_aluguel(null, calcular_numero_dias(data_inicio, data_termino));

  btnSalvarAluguel.style.display = 'block';
  btnSalvarAlteracoesAluguel.style.display = 'none';

  divModalAluguel.style.display = 'flex';
}


/*
  --------------------------------------------------------------------------------------
  Calcula o número de dias de um intervalo de duas datas
  --------------------------------------------------------------------------------------
*/
function calcular_numero_dias (data_inicio, data_termino) {
  const diffTime = Math.abs(data_inicio - data_termino);
  const dias = Math.ceil(diffTime / (1000 * 60 * 60 * 24)) + 1;
  return dias;
}


function calcular_valor_total_aluguel() {
  const veiculo = document.querySelector('option:checked').value;
  const valor_diaria = veiculo != '' ? parseFloat(document.querySelector('option:checked').dataset.valordiaria) : null;
  const data_inicio = new Date(dateDataInicio.value);
  const data_termino = new Date(dateDataTermino.value);
  const dias = calcular_numero_dias(data_inicio, data_termino);

  render_infos_aluguel(valor_diaria, dias);
}


function ler_aluguelForm() {
  const id = hiddenId.value;
  const id_cliente = vazio_vira_null(hidIdCliente.value);
  const id_veiculo = selModeloVeiculo.value != '' ? parseInt(selModeloVeiculo.value) : null;
  const data_inicio = vazio_vira_null(dateDataInicio.value);
  const data_termino = vazio_vira_null(dateDataTermino.value);

  const aluguel = {
    id: id,
    id_cliente: id_cliente,
    id_veiculo: id_veiculo,
    data_inicio: data_inicio,
    data_termino:data_termino
  };

  return aluguel;
}


function render_alugueis(alugueis) {
    let html = '';

    if (alugueis.length == 0) {
      divNaoHaAlugueis.style.display = 'block';
      divListaAlugueis_linhas.style.display = 'none';
      return;
    }

    divNaoHaAlugueis.style.display = 'none';
    divListaAlugueis_linhas.style.display = 'block';

    alugueis.forEach(i => {
        var template = divItemTemplate.innerHTML;

        const data_inicio = string_para_date(i.data_inicio);
        const data_termino = string_para_date(i.data_termino);

        html += template.
          replaceAll('[id]', i.id).
          replaceAll('[veiculo]', i.veiculo.modelo).
          replaceAll('[data_inicio]', date_para_texto_dataCurta(data_inicio)).
          replaceAll('[data_termino]', date_para_texto_dataCurta(data_termino)).
          replaceAll('[valor]', decimal_para_texto_moedareal(i.valor));
    });

    divListaAlugueis_linhas.innerHTML = html;
}


function render_infos_aluguel(valor_diaria, dias) {
  divValorDiariaVeiculo.innerHTML = valor_diaria != null ? decimal_para_texto_moedareal(valor_diaria) : '-';
  divDias.innerHTML = dias
  divTotal.innerHTML = valor_diaria != null ? decimal_para_texto_moedareal(valor_diaria * dias) : '-';
}


function valida_dados_insercao_aluguel(cliente) {

  if (selModeloVeiculo.value == '') {
    alert('Informe o veículo!');
    return false;
  }

  return true;
}