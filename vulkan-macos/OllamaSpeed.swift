// OllamaSpeed -- a menu bar readout of Ollama's generation speed.
//
// The desktop app ships as a single compiled binary with no asar bundle, so
// its UI cannot be patched, and anything that could be would be overwritten by
// the next auto-update. This runs alongside instead: the patched server logs a
// line for every completed request, and this tails it.
//
//     ollama-speed model=... prompt_tok=12 prompt_tps=84.9 gen_tok=200 gen_tps=45.7
//
// Build:  swiftc -O OllamaSpeed.swift -o OllamaSpeed
// Run:    ./OllamaSpeed
import AppKit
import Foundation

final class SpeedMonitor {
    private let path = NSHomeDirectory() + "/.ollama/logs/server.log"
    private var handle: FileHandle?
    private var offset: UInt64 = 0
    private var carry = ""

    /// Called with (generation tok/s, prompt tok/s, model) for each new entry.
    var onSample: ((Double, Double, String) -> Void)?

    func start() {
        openAndSeekToEnd()
        Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] _ in
            self?.poll()
        }
    }

    private func openAndSeekToEnd() {
        handle = FileHandle(forReadingAtPath: path)
        // Start at the end: only new generations are interesting, and the log
        // can be very large.
        if let h = handle, let end = try? h.seekToEnd() { offset = end }
    }

    private func poll() {
        guard FileManager.default.fileExists(atPath: path) else { return }

        // The log is rotated or truncated when Ollama restarts; reopen if it
        // shrank underneath us.
        let attrs = try? FileManager.default.attributesOfItem(atPath: path)
        let size = (attrs?[.size] as? NSNumber)?.uint64Value ?? 0
        if size < offset {
            handle = FileHandle(forReadingAtPath: path)
            offset = 0
            carry = ""
        }
        if handle == nil { openAndSeekToEnd(); return }

        guard let h = handle else { return }
        try? h.seek(toOffset: offset)
        guard let chunk = try? h.readToEnd(), !chunk.isEmpty else { return }
        offset += UInt64(chunk.count)

        let text = carry + (String(data: chunk, encoding: .utf8) ?? "")
        var lines = text.components(separatedBy: "\n")
        carry = lines.removeLast()          // possibly partial

        for line in lines where line.contains("ollama-speed") {
            let gen = value(in: line, key: "gen_tps")
            let prm = value(in: line, key: "prompt_tps")
            var model = string(in: line, key: "model")
            if let slash = model.lastIndex(of: "/") { model = String(model[model.index(after: slash)...]) }
            if model.count > 18 { model = String(model.prefix(18)) }
            if let g = gen { onSample?(g, prm ?? 0, model) }
        }
    }

    private func string(in line: String, key: String) -> String {
        guard let r = line.range(of: key + "=") else { return "" }
        let rest = line[r.upperBound...]
        return String(rest.prefix(while: { !$0.isWhitespace })).trimmingCharacters(in: CharacterSet(charactersIn: "\""))
    }

    private func value(in line: String, key: String) -> Double? {
        Double(string(in: line, key: key))
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    private var item: NSStatusItem!
    private let monitor = SpeedMonitor()
    private var history: [(Double, Double, String)] = []

    func applicationDidFinishLaunching(_ note: Notification) {
        item = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        setTitle("— tok/s")

        monitor.onSample = { [weak self] gen, prompt, model in
            guard let self else { return }
            DispatchQueue.main.async {
                self.history.insert((gen, prompt, model), at: 0)
                if self.history.count > 10 { self.history.removeLast() }
                self.setTitle(String(format: "%.1f tok/s", gen))
                self.rebuildMenu()
            }
        }
        monitor.start()
        rebuildMenu()
    }

    private func setTitle(_ s: String) {
        let font = NSFont.monospacedDigitSystemFont(ofSize: 12, weight: .regular)
        item.button?.attributedTitle = NSAttributedString(string: s, attributes: [.font: font])
    }

    private func rebuildMenu() {
        let menu = NSMenu()

        if history.isEmpty {
            menu.addItem(withTitle: "No generations yet", action: nil, keyEquivalent: "")
            menu.addItem(withTitle: "Run a prompt in Ollama", action: nil, keyEquivalent: "")
        } else {
            let (g, p, m) = history[0]
            menu.addItem(withTitle: String(format: "%@ — %.1f tok/s", m, g), action: nil, keyEquivalent: "")
            if p > 0 {
                menu.addItem(withTitle: String(format: "   prompt: %.1f tok/s", p), action: nil, keyEquivalent: "")
            }

            if history.count > 1 {
                menu.addItem(.separator())
                let recent = history.prefix(8).map { $0.0 }
                let avg = recent.reduce(0, +) / Double(recent.count)
                menu.addItem(withTitle: String(format: "avg of last %d: %.1f tok/s", recent.count, avg),
                             action: nil, keyEquivalent: "")
                // Sustained load on this GPU throttles hard; show the trend.
                if let first = recent.last, let latest = recent.first, first > 0 {
                    let delta = 100 * (latest - first) / first
                    if abs(delta) >= 5 {
                        menu.addItem(withTitle: String(format: "trend: %+.0f%% since oldest shown", delta),
                                     action: nil, keyEquivalent: "")
                    }
                }
                menu.addItem(.separator())
                for (i, h) in history.enumerated().dropFirst() {
                    menu.addItem(withTitle: String(format: "%2d.  %.1f tok/s   %@", i, h.0, h.2),
                                 action: nil, keyEquivalent: "")
                }
            }
        }

        menu.addItem(.separator())
        menu.addItem(withTitle: "Quit", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        item.menu = menu
    }
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.setActivationPolicy(.accessory)   // menu bar only, no dock icon
app.run()
