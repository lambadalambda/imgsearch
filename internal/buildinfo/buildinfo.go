// Package buildinfo reports the version, commit, and build date of the
// running binary. Release builds stamp the variables with -ldflags -X;
// development builds fall back to the VCS metadata Go embeds.
package buildinfo

import (
	"fmt"
	"runtime/debug"
	"strings"
)

// Set at link time, for example:
//
//	-ldflags "-X imgsearch/internal/buildinfo.Version=rolling-20260921-abc123 ..."
var (
	Version = ""
	Commit  = ""
	Date    = ""
)

// Info is the JSON shape exposed by the API.
type Info struct {
	Version string `json:"version"`
	Commit  string `json:"commit"`
	Date    string `json:"date"`
}

// Current returns the stamped values, filling gaps from Go's embedded VCS
// metadata when available.
func Current() Info {
	return resolve(Version, Commit, Date, debug.ReadBuildInfo)
}

func resolve(version string, commit string, date string, read func() (*debug.BuildInfo, bool)) Info {
	info := Info{Version: strings.TrimSpace(version), Commit: strings.TrimSpace(commit), Date: strings.TrimSpace(date)}
	if info.Commit == "" || info.Date == "" {
		if bi, ok := read(); ok && bi != nil {
			for _, setting := range bi.Settings {
				switch setting.Key {
				case "vcs.revision":
					if info.Commit == "" {
						info.Commit = setting.Value
					}
				case "vcs.time":
					if info.Date == "" {
						info.Date = setting.Value
					}
				case "vcs.modified":
					if setting.Value == "true" && info.Commit != "" && !strings.HasSuffix(info.Commit, "-dirty") {
						info.Commit += "-dirty"
					}
				}
			}
		}
	}
	if info.Version == "" {
		info.Version = "dev"
	}
	return info
}

// String renders a one-line description for -version output.
func (i Info) String() string {
	commit := i.Commit
	if commit == "" {
		commit = "unknown"
	}
	date := i.Date
	if date == "" {
		date = "unknown"
	}
	return fmt.Sprintf("imgsearch %s (commit %s, built %s)", i.Version, commit, date)
}
