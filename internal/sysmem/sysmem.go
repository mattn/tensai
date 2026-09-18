// Package sysmem answers one question: how much memory a model may
// take. What "available" means differs by platform, and a platform
// that cannot say answers zero, which callers read as unknown.
package sysmem
