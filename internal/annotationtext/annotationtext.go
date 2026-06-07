package annotationtext

import "strings"

const fallbackTitleMaxChars = 160

type Fields struct {
	Title           string
	Summary         string
	Description     string
	FullDescription string
}

func Build(title string, summary string, fullDescription string) Fields {
	full := strings.TrimSpace(fullDescription)
	summary = strings.TrimSpace(summary)
	if full == "" {
		full = summary
	}
	if summary == "" {
		summary = full
	}
	title = strings.TrimSpace(title)
	if title == "" {
		title = DeriveTitle(full)
	}
	return Fields{
		Title:           title,
		Summary:         summary,
		Description:     summary,
		FullDescription: full,
	}
}

func DeriveTitle(description string) string {
	cleaned := collapseWhitespace(description)
	if cleaned == "" {
		return ""
	}
	first := firstSentence(cleaned)
	runes := []rune(first)
	if len(runes) <= fallbackTitleMaxChars {
		return first
	}
	return strings.TrimSpace(string(runes[:fallbackTitleMaxChars-3])) + "..."
}

func firstSentence(text string) string {
	for i := 0; i < len(text); i++ {
		switch text[i] {
		case '.', '!', '?':
			if i+1 == len(text) || text[i+1] == ' ' {
				return strings.TrimSpace(text[:i+1])
			}
		}
	}
	return strings.TrimSpace(text)
}

func collapseWhitespace(input string) string {
	return strings.Join(strings.Fields(strings.TrimSpace(input)), " ")
}
