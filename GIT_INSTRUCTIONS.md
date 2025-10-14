# 📝 Instruções de Controle de Versão - Git

Este documento explica como versionar o projeto K-Means 1D no Git.

---

## 🚀 Inicialização do Repositório

### Primeira vez (criar repositório local)
```bash
# Navegar até o diretório do projeto
cd "d:\Projetinhos\Faculdade\PCD\Entrega 1"

# Inicializar repositório Git
git init

# Adicionar arquivos essenciais
git add .gitignore
git add *.c *.py *.ps1 *.md

# Primeiro commit
git commit -m "Inicial: K-Means 1D com OpenMP - versão otimizada"
```

---

## 📦 O que está sendo Versionado

### ✅ Incluído no Repositório
- **Código fonte:** `*.c`, `*.py`, `*.ps1`
- **Documentação:** `*.md` (README, relatórios, etc.)
- **Scripts:** Scripts de automação e geração
- **Configuração:** `.gitignore`, `prompt.md`
- **Centróides:** `centroides_iniciais.csv`, `centroids_*.csv` (pequenos)
- **Imagens:** `*.png` (gráficos de resultados)

### ❌ Excluído do Repositório
- **Executáveis:** `*.exe`, `*.dll`
- **Dados grandes:** `dados.csv` (~75 MB)
- **Resultados grandes:** `assign_*.csv` (~15 MB cada)
- **Arquivos temporários:** `*.o`, `*.tmp`, `*.log`
- **Configurações IDE:** `.vscode/`, `.idea/`
- **Sistema:** `.DS_Store`, `Thumbs.db`

---

## 📋 Workflows Comuns

### Adicionar Novas Modificações
```bash
# Ver status dos arquivos modificados
git status

# Adicionar arquivos específicos
git add kmeans_1d_omp.c
git add RELATORIO_FINAL.md

# Ou adicionar todos os arquivos modificados
git add -u

# Commit com mensagem descritiva
git commit -m "Otimização: Adiciona padding para eliminar false sharing"
```

### Verificar Histórico
```bash
# Ver log de commits
git log --oneline

# Ver diferenças não commitadas
git diff

# Ver diferenças de um arquivo específico
git diff kmeans_1d_omp.c
```

### Branches para Desenvolvimento
```bash
# Criar branch para nova feature
git checkout -b feature/gpu-support

# Voltar para branch principal
git checkout main

# Listar branches
git branch -a

# Deletar branch
git branch -d feature/gpu-support
```

---

## 🌐 Conectar com GitHub/GitLab

### Criar Repositório Remoto
1. Criar repositório no GitHub: https://github.com/new
2. Nome sugerido: `kmeans-1d-openmp`
3. Descrição: "K-Means 1D clustering com paralelização OpenMP"
4. Visibilidade: Privado (se for trabalho acadêmico)

### Conectar ao Remoto
```bash
# Adicionar remote
git remote add origin https://github.com/SEU_USUARIO/kmeans-1d-openmp.git

# Renomear branch principal para main (se necessário)
git branch -M main

# Enviar para o remoto
git push -u origin main
```

### Operações com Remoto
```bash
# Baixar mudanças
git pull origin main

# Enviar mudanças
git push origin main

# Ver remotes configurados
git remote -v
```

---

## 📊 Versionando Dados Grandes

### Opção 1: Git LFS (Large File Storage)
Para versionar dados grandes (dados.csv):

```bash
# Instalar Git LFS
git lfs install

# Rastrear arquivos grandes
git lfs track "*.csv"
git lfs track "dados.csv"

# Adicionar .gitattributes
git add .gitattributes

# Agora pode adicionar arquivos grandes
git add dados.csv
git commit -m "Adiciona dataset com Git LFS"
```

### Opção 2: Script de Regeneração
**Recomendado:** Não versionar dados grandes, apenas script de geração:

```bash
# Adicionar apenas o gerador
git add generate_data.py

# Documentar no README como regenerar
echo "Execute: python generate_data.py 5000000 20 42" >> README.md
```

---

## 🏷️ Tags e Releases

### Marcar Versões
```bash
# Tag para entrega
git tag -a v1.0-entrega -m "Versão final para entrega - Otimizada"

# Tag para versão intermediária
git tag -a v0.1-inicial -m "Versão inicial (speedup negativo)"

# Listar tags
git tag -l

# Enviar tag para remoto
git push origin v1.0-entrega
```

---

## 🔄 Sincronização em Equipe

### Workflow de Colaboração
```bash
# Antes de começar a trabalhar
git pull origin main

# Fazer suas modificações
# ... editar arquivos ...

# Adicionar e commitar
git add .
git commit -m "Descrição da mudança"

# Enviar para o remoto
git push origin main
```

### Resolver Conflitos
```bash
# Se houver conflitos ao fazer pull
git pull origin main

# Editar arquivos com conflitos
# Procurar por: <<<<<<< HEAD, =======, >>>>>>>

# Após resolver
git add arquivo_resolvido.c
git commit -m "Resolve conflitos de merge"
git push origin main
```

---

## 📜 Mensagens de Commit - Boas Práticas

### Formato Recomendado
```
<tipo>: <descrição curta>

<descrição detalhada (opcional)>

<referências (opcional)>
```

### Tipos de Commit
- **feat:** Nova funcionalidade
- **fix:** Correção de bug
- **perf:** Melhoria de performance
- **refactor:** Refatoração de código
- **docs:** Documentação
- **test:** Testes
- **chore:** Tarefas gerais

### Exemplos
```bash
git commit -m "feat: Adiciona paralelização do update step"

git commit -m "perf: Otimiza chunk size para 10000"

git commit -m "fix: Corrige false sharing com padding"

git commit -m "docs: Atualiza README com resultados finais"
```

---

## 🔍 Comandos Úteis

### Inspeção
```bash
# Ver tamanho do repositório
git count-objects -vH

# Ver quem modificou cada linha
git blame kmeans_1d_omp.c

# Buscar no histórico
git log --all --grep="speedup"

# Ver commit específico
git show <commit-hash>
```

### Limpeza
```bash
# Remover arquivos não rastreados
git clean -n  # preview
git clean -f  # executar

# Descartar mudanças locais
git checkout -- kmeans_1d_omp.c

# Reset completo (CUIDADO!)
git reset --hard HEAD
```

### Desfazer
```bash
# Desfazer último commit (mantém mudanças)
git reset --soft HEAD~1

# Desfazer último commit (descarta mudanças)
git reset --hard HEAD~1

# Reverter commit específico
git revert <commit-hash>
```

---

## 📚 Estrutura Recomendada de Commits

### Sugestão de Histórico
```
v1.0-entrega (tag)
├── docs: Adiciona relatório final e sumário
├── perf: Otimiza compilação com -O3 -march=native
├── perf: Adiciona padding para eliminar false sharing
├── perf: Otimiza chunk size para 10000
├── feat: Aumenta dataset para 5M pontos e 20 clusters
├── docs: Atualiza documentação com novos resultados
├── fix: Corrige speedup negativo
v0.1-inicial (tag)
├── docs: Adiciona relatório de experimentos
├── test: Adiciona script de comparação de resultados
├── feat: Adiciona visualização de resultados
├── feat: Implementa versão paralela OpenMP
├── feat: Implementa versão serial
├── chore: Adiciona scripts de automação
└── Initial commit
```

---

## 🔐 Segurança

### Não Versionar
- ❌ Senhas ou tokens
- ❌ Chaves SSH/API
- ❌ Dados pessoais sensíveis
- ❌ Configurações locais de IDE

### Remover Arquivo Acidentalmente Commitado
```bash
# Remover do Git mas manter no disco
git rm --cached arquivo_sensivel.txt
git commit -m "Remove arquivo sensível"

# Adicionar ao .gitignore
echo "arquivo_sensivel.txt" >> .gitignore
git add .gitignore
git commit -m "Adiciona arquivo ao .gitignore"
```

---

## 📖 Recursos Adicionais

- **Git Cheat Sheet:** https://education.github.com/git-cheat-sheet-education.pdf
- **Pro Git Book:** https://git-scm.com/book/pt-br/v2
- **GitHub Docs:** https://docs.github.com/
- **Git LFS:** https://git-lfs.github.com/

---

## ✅ Checklist Final

Antes de fazer push:
- [ ] Código compila sem erros
- [ ] Testes passam
- [ ] Documentação atualizada
- [ ] .gitignore configurado
- [ ] Mensagens de commit descritivas
- [ ] Sem arquivos sensíveis ou grandes desnecessários

---

**Data:** 14 de Outubro de 2025  
**Projeto:** K-Means 1D com OpenMP  
**Status:** Pronto para versionamento
