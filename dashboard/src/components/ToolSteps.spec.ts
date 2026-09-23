import { describe, expect, it } from "vitest";
import { mount } from "../tests/mount";
import ToolSteps from "./ToolSteps.vue";

describe("ToolSteps", () => {
    it("renders nothing without steps", () => {
        const wrapper = mount(ToolSteps, { props: { steps: [] } });
        expect(wrapper.find('[data-testid="tool-steps"]').exists()).toBe(false);
    });

    it("shows the tool name, truncated args, and a full-text tooltip", () => {
        const args = { query: "x".repeat(80) };
        const wrapper = mount(ToolSteps, {
            props: {
                steps: [
                    { name: "search_documents", arguments: args, result: null },
                ],
            },
        });
        expect(wrapper.find('[data-testid="tool-step-name"]').text()).toBe(
            "search_documents",
        );
        const argsEl = wrapper.find('[data-testid="tool-step-args"]');
        expect(argsEl.attributes("title")).toBe(JSON.stringify(args));
        // Truncated with an ellipsis, not the full 90-char serialization.
        expect(argsEl.text()).toBe(JSON.stringify(args).slice(0, 60) + "…");
        expect(
            wrapper.find('[data-testid="tool-step-result-toggle"]').exists(),
        ).toBe(false);
        // A pending step shows its spinner.
        expect(wrapper.find("svg.spin-icon").exists()).toBe(true);
    });

    it("expands a tool result on click", async () => {
        const wrapper = mount(ToolSteps, {
            props: {
                steps: [
                    { name: "sql", arguments: null, result: "the full result" },
                ],
            },
        });
        expect(wrapper.find('[data-testid="tool-step-result"]').exists()).toBe(
            false,
        );
        await wrapper
            .find('[data-testid="tool-step-result-toggle"]')
            .trigger("click");
        expect(wrapper.find('[data-testid="tool-step-result"]').text()).toBe(
            "the full result",
        );
    });

    it("collapses the whole activity block", async () => {
        const wrapper = mount(ToolSteps, {
            props: {
                steps: [{ name: "sql", arguments: null, result: "r" }],
            },
        });
        expect(wrapper.find('[data-testid="tool-step-name"]').exists()).toBe(
            true,
        );
        await wrapper
            .find('[data-testid="tool-steps-toggle"]')
            .trigger("click");
        expect(wrapper.find('[data-testid="tool-step-name"]').exists()).toBe(
            false,
        );
    });
});
