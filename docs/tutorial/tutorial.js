// The ferrompi tutorial's one script: colors the code blocks and gives each a
// Copy button. Pages read the same without it; the code stays plain text in
// the HTML, which is what the CI snippet check extracts.
(() => {
  "use strict";

  // One alternation per language. The capture group that matched picks the
  // token class; text between matches stays uncolored.
  const LANGUAGES = {
    rust: {
      pattern:
        /(\/\/[^\n]*)|("(?:\\.|[^"\\])*")|('(?:\\.|[^'\\])')|(#!?\[[^\]\n]*\])|(\b[a-z_][a-z0-9_]*!)|(\b(?:as|break|const|continue|crate|dyn|else|enum|extern|false|fn|for|if|impl|in|let|loop|match|mod|move|mut|pub|ref|return|self|Self|static|struct|super|trait|true|type|unsafe|use|where|while)\b)|(\b[A-Z][A-Za-z0-9_]*\b)|(\b\d[\d_]*(?:\.\d+)?\b)/g,
      classes: [null, "com", "str", "str", "mac", "mac", "kw", "ty", "num"],
    },
    bash: {
      pattern: /(#[^\n]*)|("(?:\\.|[^"\\])*"|'[^']*')|(\$\{?[A-Za-z_][A-Za-z0-9_]*\}?)/g,
      classes: [null, "com", "str", "mac"],
    },
    // MPI's and ferrompi's names take the type color, whether they name a
    // type, a constant or a function.
    c: {
      pattern:
        /(\/\*[\s\S]*?\*\/|\/\/[^\n]*)|("(?:\\.|[^"\\])*")|(^[ \t]*#[ \t]*[a-z]+)|(\b(?:break|case|char|const|continue|default|defined|do|double|else|enum|extern|float|for|if|int|long|return|short|signed|sizeof|static|struct|switch|typedef|union|unsigned|void|while)\b)|(\b(?:MPI|OMPI|FERROMPI)_[A-Za-z0-9_]*\b|\b[a-z][a-z0-9_]*_t\b)|(\b(?:0x[0-9a-fA-F]+|\d+)\b)/gm,
      classes: [null, "com", "str", "mac", "kw", "ty", "num"],
    },
  };

  const escapeHtml = (text) =>
    text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

  function highlight(code, { pattern, classes }) {
    const text = code.textContent;
    let html = "";
    let last = 0;
    for (const match of text.matchAll(pattern)) {
      const group = match.findIndex((g, i) => i > 0 && g !== undefined);
      html += escapeHtml(text.slice(last, match.index));
      html += `<span class="tok-${classes[group]}">${escapeHtml(match[0])}</span>`;
      last = match.index + match[0].length;
    }
    code.innerHTML = html + escapeHtml(text.slice(last));
  }

  function addCopyButton(pre, code) {
    const wrapper = document.createElement("div");
    wrapper.className = "code-block";
    pre.replaceWith(wrapper);
    wrapper.append(pre);

    const button = document.createElement("button");
    button.type = "button";
    button.className = "copy";
    button.textContent = "Copy";
    button.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(code.textContent);
        button.textContent = "Copied";
      } catch {
        const range = document.createRange();
        range.selectNodeContents(code);
        const selection = window.getSelection();
        selection.removeAllRanges();
        selection.addRange(range);
        button.textContent = "Selected";
      }
      setTimeout(() => {
        button.textContent = "Copy";
      }, 1500);
    });
    wrapper.append(button);
  }

  for (const pre of document.querySelectorAll("pre[data-lang]")) {
    const code = pre.querySelector("code");
    if (!code) continue;
    const language = LANGUAGES[pre.dataset.lang];
    if (language) highlight(code, language);
    addCopyButton(pre, code);
  }
})();
