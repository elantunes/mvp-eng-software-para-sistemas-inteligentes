/*
  --------------------------------------------------------------------------------------
  Obtém um String contendo uma data em formato curto.
  --------------------------------------------------------------------------------------
*/
function date_para_texto_dataCurta(data) {
    const retorno = data.toLocaleDateString('pt-br');
    return retorno;
}


/*
  --------------------------------------------------------------------------------------
  Obtém um String em formato de moeda em real. Ex. (R$ 4.940,30).
  --------------------------------------------------------------------------------------
*/
function decimal_para_texto_moedareal(decimal) {
  const retorno = decimal.toLocaleString('pt-br',{style: 'currency', currency: 'BRL'});
  return retorno;
}


/*
  --------------------------------------------------------------------------------------
  Obtém um date a partir de uma string. Importante para cancelar mudanças vindas de 
  fusos horários.
  --------------------------------------------------------------------------------------
*/
function string_para_date(data) {
  const retorno = new Date(new Date(data).toISOString().slice(0,-1));
  return retorno;
}


/*
  --------------------------------------------------------------------------------------
  Indicado para validações de formulários.
  --------------------------------------------------------------------------------------
*/
function vazio_vira_null(texto) {
  resultado = texto != '' ? texto : null;
  return resultado;
}
