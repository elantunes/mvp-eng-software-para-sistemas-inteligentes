SERVIDOR_API = 'http://localhost:5001/';
SERVIDOR_API_ANUNCIOS = 'http://localhost:5002/';


btnPublicar.onclick = function() {

  let anuncios = [];

  Array.from(lista_anuncios.children).forEach(function(x) {
    const anuncio = new Anuncio(
        parseInt(x.dataset["id_veiculo"]),
        x.dataset["modelo"],
        parseFloat(x.dataset["valor_diaria"]));

    anuncios.push(anuncio);
  });

  btnPublicar.disabled = 'disabled';

  const formData = new FormData();
  formData.append('anuncios',JSON.stringify(anuncios))

  const url = `${SERVIDOR_API_ANUNCIOS}anuncios`;
  fetch(url, {
    method: 'post',
    body: formData
  })
  .then((response) => response.json())
  .then(function() {
    ativar_popup_operacao_com_sucesso();
  })
  .catch((error) => console.error('Error:', error));

  btnPublicar.disabled = '';
};


$('#lista_anuncios').on('click','div[data-veiculo-item]', function(e) {
  const ordem = parseInt(this.dataset["ordem"]);
  const veiculosNaoSelecionados = document.querySelectorAll('#lista_veiculos div[data-ordem]')

  const divAnterior =
    Array
    .from(veiculosNaoSelecionados)
    .filter(x => parseInt(x.dataset['ordem']) < ordem)
    .pop()

  if (divAnterior != undefined)
      divAnterior.after(this);
  else
    lista_veiculos.insertBefore(this, lista_veiculos.firstChild);

  render_veiculos_anuncios_sem_dados(lista_anuncios.children);
});


$('#lista_veiculos').on('click','div[data-veiculo-item]', function(e) {
  lista_anuncios.append(this);
  render_veiculos_anuncios_sem_dados(lista_anuncios.children);
});


/*
  --------------------------------------------------------------------------------------
  Obtém uma lista de anúncios via requisição GET
  --------------------------------------------------------------------------------------
*/
const get_anuncios = async () => {
    const url = `${SERVIDOR_API}veiculos`;
    let veiculos;

    fetch(url, {
      method: 'get',
    })
    .then((response) => response.json())
    .then(function(data) {
      veiculos = data.veiculos;
      const url = `${SERVIDOR_API_ANUNCIOS}anuncios`;
      fetch(url, {
        method: 'get',
      })
      .then((response) => response.json())
      .then(function(data) {
        const idsVeiculosAnunciados = data.anuncios.map((x) => x.id_veiculo);
        const veiculosAnunciados = data.anuncios;
        const veiculosNaoAnunciados = veiculos.filter(x => !idsVeiculosAnunciados.includes(x.id_veiculo));
        render_veiculos(lista_anuncios, veiculosAnunciados, veiculos);
        render_veiculos(lista_veiculos, veiculosNaoAnunciados, veiculos);
        render_veiculos_anuncios_sem_dados(data.anuncios);

        secAnunciosCarregando.style.display = 'none';
        secAnunciosSucesso.style.display = 'block';
      })
      .catch(function(error) {
        console.error('Error:', error);
        secAnunciosCarregando.style.display = 'none';
        secAnunciosErro.style.display = 'block';
      });

    })
    .catch(function(error) {
      secAnunciosCarregando.style.display = 'none';
      secAnunciosErro.style.display = 'block';
      console.error('Error:', error);
    });
}


function render_veiculos(elemento, veiculosFiltrados, veiculos) {
  let html = '';

  veiculosFiltrados.forEach(i => {
    let template = template_card_veiculo.innerHTML;
    const ordem = veiculos.findIndex(x => x.id_veiculo == i.id_veiculo);
    html +=
      template.
      replaceAll('[id_veiculo]', i.id_veiculo).
      replaceAll('[modelo]', i.modelo).
      replaceAll('[valor_diaria]', i.valor_diaria.toLocaleString()).
      replaceAll('[ordem]', ordem);
  });

  elemento.innerHTML = html;
}


function render_veiculos_anuncios_sem_dados(anuncios) {
  const temDados = anuncios.length > 0;
  lista_anuncios_sem_dados.style.display = !temDados ? 'block' : 'none';
  btnPublicar.disabled = !temDados ? 'disabled' : '';
}
