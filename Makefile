LOCAL_BIN:=$(CURDIR)/bin

# Check global GOLANGCI-LINT
GOLANGCI_BIN:=$(LOCAL_BIN)/golangci-lint
GOLANGCI_TAG:=1.13.1

# install golangci-lint binary
.PHONY: install-lint
install-lint:
ifeq ($(wildcard $(GOLANGCI_BIN)),)
	$(info #Downloading golangci-lint v$(GOLANGCI_TAG))
	go get -d github.com/golangci/golangci-lint@v$(GOLANGCI_TAG)
	go build -ldflags "-X 'main.version=$(GOLANGCI_TAG)' -X 'main.commit=test' -X 'main.date=test'" -o $(LOCAL_BIN)/golangci-lint github.com/golangci/golangci-lint/cmd/golangci-lint
	go mod tidy
GOLANGCI_BIN:=$(LOCAL_BIN)/golangci-lint
endif

# run diff lint like in pipeline
.PHONY: .lint
.lint: install-lint
	$(info #Running lint...)
	$(GOLANGCI_BIN) run --new-from-rev=origin/master --config=.golangci.yaml ./...

# golangci-lint diff master
.PHONY: lint
lint: .lint

# run full lint like in pipeline
.PHONY: lint-full
lint-full: install-lint
	$(GOLANGCI_BIN) run --config=.golangci.yaml ./...