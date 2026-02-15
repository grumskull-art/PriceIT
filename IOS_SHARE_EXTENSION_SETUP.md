# iOS Share Extension Setup (PriceIT)

This repo contains `extension_preprocessing.js`, which extracts product data from Safari pages.
This guide wires that script to an iOS Share Extension and sends the payload to `POST /v1/analyze`.

## 1. Add Share Extension target in Xcode
1. Open your iOS app project.
2. `File` -> `New` -> `Target...` -> `Share Extension`.
3. Name it, for example: `PriceITShareExtension`.
4. Choose Swift.

## 2. Add preprocessing JavaScript file
1. Copy `extension_preprocessing.js` into the Share Extension target folder.
2. In Xcode File Inspector, ensure target membership is checked for the Share Extension.

## 3. Configure Share Extension `Info.plist`
Set these keys in the Share Extension target plist:

```xml
<key>NSExtension</key>
<dict>
  <key>NSExtensionAttributes</key>
  <dict>
    <key>NSExtensionActivationRule</key>
    <dict>
      <key>NSExtensionActivationSupportsWebURLWithMaxCount</key>
      <integer>1</integer>
    </dict>
    <key>NSExtensionJavaScriptPreprocessingFile</key>
    <string>extension_preprocessing</string>
  </dict>
  <key>NSExtensionMainStoryboard</key>
  <string>MainInterface</string>
  <key>NSExtensionPointIdentifier</key>
  <string>com.apple.share-services</string>
</dict>

<key>PRICEIT_API_BASE_URL</key>
<string>http://192.168.1.50:8000</string>

<key>PRICEIT_API_KEY</key>
<string>REPLACE_WITH_API_KEY</string>
```

Notes:
- `NSExtensionJavaScriptPreprocessingFile` must be filename without `.js`.
- For local testing, use your computer LAN IP (same Wi-Fi as phone), not `localhost`.

## 4. ATS for local HTTP (if backend is not HTTPS)
If you call `http://...`, add this in the Share Extension plist:

```xml
<key>NSAppTransportSecurity</key>
<dict>
  <key>NSAllowsArbitraryLoads</key>
  <true/>
</dict>
```

For production, switch to HTTPS and remove broad ATS exceptions.

## 5. If backend runs inside WSL (important)
When FastAPI runs in WSL2, iPhone usually cannot reach it directly unless Windows forwards the port.

Run these in **Windows PowerShell as Administrator**:

```powershell
# Find current WSL IP
wsl hostname -I

# Replace <WSL_IP> with the value above (first IP)
netsh interface portproxy add v4tov4 listenaddress=0.0.0.0 listenport=8000 connectaddress=<WSL_IP> connectport=8000

# Open firewall for port 8000
netsh advfirewall firewall add rule name="PriceIT 8000" dir=in action=allow protocol=TCP localport=8000
```

Then set:
- `PRICEIT_API_BASE_URL` = `http://<WINDOWS_LAN_IP>:8000`

Notes:
- WSL IP can change after reboot; update `portproxy` when it changes.
- Check current proxy rules with: `netsh interface portproxy show all`

## 6. Replace ShareViewController with this Swift code
Use this in the Share Extension target `ShareViewController.swift`:

```swift
import Social
import UIKit
import UniformTypeIdentifiers

final class ShareViewController: SLComposeServiceViewController {
    private let propertyListType = UTType.propertyList.identifier

    override func isContentValid() -> Bool {
        true
    }

    override func didSelectPost() {
        loadPayload { [weak self] payload in
            guard let self = self else { return }
            guard var payload = payload else {
                self.cancelWithError("Could not extract product data from page.")
                return
            }

            // Default values used by backend if caller does not supply them.
            if payload["country"] == nil {
                payload["country"] = "dk"
            }
            if payload["currency"] == nil {
                payload["currency"] = "DKK"
            }

            self.postAnalyze(payload: payload)
        }
    }

    override func configurationItems() -> [Any]! {
        []
    }

    private func loadPayload(completion: @escaping ([String: Any]?) -> Void) {
        guard let inputItems = extensionContext?.inputItems as? [NSExtensionItem] else {
            completion(nil)
            return
        }

        for item in inputItems {
            guard let attachments = item.attachments else { continue }
            for provider in attachments where provider.hasItemConformingToTypeIdentifier(propertyListType) {
                provider.loadItem(forTypeIdentifier: propertyListType, options: nil) { item, _ in
                    guard
                        let dict = item as? NSDictionary,
                        let results = dict[NSExtensionJavaScriptPreprocessingResultsKey] as? [String: Any]
                    else {
                        completion(nil)
                        return
                    }
                    completion(results)
                }
                return
            }
        }

        completion(nil)
    }

    private func postAnalyze(payload: [String: Any]) {
        guard
            let base = Bundle.main.object(forInfoDictionaryKey: "PRICEIT_API_BASE_URL") as? String,
            let apiKey = Bundle.main.object(forInfoDictionaryKey: "PRICEIT_API_KEY") as? String,
            let url = URL(string: base + "/v1/analyze")
        else {
            cancelWithError("Missing PRICEIT_API_BASE_URL or PRICEIT_API_KEY in Info.plist.")
            return
        }

        let sanitizedPayload = payload.filter { !($0.value is NSNull) }
        guard let body = try? JSONSerialization.data(withJSONObject: sanitizedPayload, options: []) else {
            cancelWithError("Could not serialize payload.")
            return
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue(apiKey, forHTTPHeaderField: "X-API-Key")
        request.httpBody = body
        request.timeoutInterval = 15

        URLSession.shared.dataTask(with: request) { [weak self] data, response, error in
            DispatchQueue.main.async {
                if let error = error {
                    self?.cancelWithError("Network error: \(error.localizedDescription)")
                    return
                }

                guard let http = response as? HTTPURLResponse else {
                    self?.cancelWithError("No HTTP response.")
                    return
                }

                guard (200...299).contains(http.statusCode) else {
                    let message: String
                    if let data = data, let body = String(data: data, encoding: .utf8), !body.isEmpty {
                        message = "API error \(http.statusCode): \(body)"
                    } else {
                        message = "API error \(http.statusCode)."
                    }
                    self?.cancelWithError(message)
                    return
                }

                self?.extensionContext?.completeRequest(returningItems: [], completionHandler: nil)
            }
        }.resume()
    }

    private func cancelWithError(_ message: String) {
        let error = NSError(domain: "PriceIT.ShareExtension", code: 1, userInfo: [
            NSLocalizedDescriptionKey: message
        ])
        extensionContext?.cancelRequest(withError: error)
    }
}
```

## 7. Verify quickly
1. Ensure backend is running: `http://<LAN_IP>:8000/health` returns `{"status":"ok"}`.
2. On iPhone Safari, open a product page.
3. Tap share sheet -> your Share Extension.
4. Tap `Post`.

If it fails:
- Check device + computer on same network.
- Confirm `PRICEIT_API_BASE_URL` and `PRICEIT_API_KEY`.
- Confirm ATS key exists for local `http://`.
- Confirm `extension_preprocessing.js` is included in extension target membership.
