SERVIDOR_API = 'http://localhost:5000/';

btnPredicaoPaciente.onclick = function() {
    const dados = ler_pacienteForm();
    post_paciente(dados);
  };

/*
  --------------------------------------------------------------------------------------
  Realiza a predição
  --------------------------------------------------------------------------------------
*/
const post_paciente = async (paciente) => {
    const url = `${SERVIDOR_API}predicao`;
  
    const formData = new FormData();
    formData.append('idade', paciente.idade);
    formData.append('altura', paciente.altura);
    formData.append('peso', paciente.peso);
    formData.append('cintura', paciente.cintura);
    formData.append('visao_esquerda', paciente.visao_esquerda);
    formData.append('visao_direita', paciente.visao_direita);
    formData.append('audicao_esquerda', paciente.audicao_esquerda);
    formData.append('audicao_direita', paciente.audicao_direita);
    formData.append('sistolica', paciente.sistolica);
    formData.append('relaxado', paciente.relaxado);
    formData.append('acucar_no_sangue_em_jejum', paciente.acucar_no_sangue_em_jejum);
    formData.append('colesterol', paciente.colesterol);
    formData.append('trigliceridos', paciente.trigliceridos);
    formData.append('HDL', paciente.HDL);
    formData.append('LDL', paciente.LDL);
    formData.append('hemoglobina', paciente.hemoglobina);
    formData.append('proteina_na_urina', paciente.proteina_na_urina);
    formData.append('creatinina_serica', paciente.creatinina_serica);
    formData.append('AST', paciente.AST);
    formData.append('ALT', paciente.ALT);
    formData.append('Gtp', paciente.Gtp);
    formData.append('caries_dentarias', paciente.caries_dentarias);
    formData.append('tartaro', paciente.tartaro);

    fetch(url, {
      method: 'post',
      body: formData
    })
    .then((response) => response.json())
    .then(function(data) {
        if (data.length == undefined) {
            if (data.fumante)
                alert('O paciente é um POSSÍVEL FUMANTE');
            else
                alert('O paciente possivelmente NÃO É FUMANTE');
        }
        else
            alert('Ocorreu um erro. Confira os campos preenchidos e tente novamente');
    })
    .catch((error) => {
        debugger;
        console.error('Error:', error)
    });
  }

  function ler_pacienteForm() {
    
    const idade = parseInt(txtIdade.value);
    const altura = parseInt(txtAltura.value);
    const peso = parseInt(txtPeso.value);
    const cintura = parseFloat(txtCintura.value);
    const visao_esquerda = parseFloat(txtVisao_esquerda.value);
    const visao_direita = parseFloat(txtVisao_direita.value);
    const audicao_esquerda = parseFloat(txtAudicao_esquerda.value);
    const audicao_direita = parseFloat(txtAudicao_direita.value);
    const sistolica = parseFloat(txtSistolica.value);
    const relaxado = parseFloat(txtRelaxado.value);
    const acucar_no_sangue_em_jejum = parseFloat(txtAcucar_no_sangue_em_jejum.value);
    const colesterol = parseFloat(txtColesterol.value);
    const trigliceridos = parseFloat(txtTrigliceridos.value);
    const HDL = parseFloat(txtHDL.value);
    const LDL = parseFloat(txtLDL.value);
    const hemoglobina = parseFloat(txtHemoglobina.value);
    const proteina_na_urina = parseFloat(txtProteina_na_urina.value);
    const creatinina_serica = parseFloat(txtCreatinina_serica.value);
    const AST = parseFloat(txtAST.value);
    const ALT = parseFloat(txtALT.value);
    const Gtp = parseFloat(txtGtp.value);
    const caries_dentarias = cbxCaries_dentarias.checked;
    const tartaro = cbxTartaro.checked;
  
    const paciente = {
        idade: idade,
        altura: altura,
        peso: peso,
        cintura: cintura,
        visao_esquerda: visao_esquerda,
        visao_direita: visao_direita,
        audicao_esquerda: audicao_esquerda,
        audicao_direita: audicao_direita,
        sistolica: sistolica,
        relaxado: relaxado,
        acucar_no_sangue_em_jejum: acucar_no_sangue_em_jejum,
        colesterol: colesterol,
        trigliceridos: trigliceridos,
        HDL: HDL,
        LDL: LDL,
        hemoglobina: hemoglobina,
        proteina_na_urina: proteina_na_urina,
        creatinina_serica: creatinina_serica,
        AST: AST,
        ALT: ALT,
        Gtp: Gtp,
        caries_dentarias: caries_dentarias,
        tartaro: tartaro
    };
  
    return paciente;
  }